# Luminal Core — Canonical Reverse-Engineering Guide

> **Purpose.** This document reverse-engineers the *core* of the Luminal ML
> compiler (the `luminal-ai/luminal` fork cloned at `refs/luminal`, branch tip
> at clone time) to the depth needed to **fork it and swap in CuTe as the
> compiler IR**. It deliberately ignores wrapper/ergonomics code and focuses on
> the three machines the user must understand with zero ambiguity: the
> **e-graph**, the **search algorithm**, and the **lowering of front-end API
> calls into the op set**.
>
> **Citation convention.** Every `path:line` citation is relative to the luminal
> repo root (`refs/luminal/`). Claims are verified against source read at clone
> time; where a claim is inferred from a name or schedule position rather than
> read line-by-line, it is marked *(inferred)*.
>
> **Structure.** §0 is high-level architectural insight (no minutiae). **Part I**
> is a self-contained mathematical theory of the system. **Part II** is the
> implementation, citing code and referencing Part I. **Part III** is the
> concrete guide to forking onto a CuTe IR. Parts I and II are segregated on
> purpose: read Part I to understand *what* the machinery computes; read Part II
> to see *where and how*.

---

## 0. Orientation — the one-page mental model

Luminal is an **equality-saturation compiler**. Unlike a classic pass-pipeline
compiler (lower, then fuse, then schedule, then allocate — each pass destroying
the previous form), Luminal represents the program as an **e-graph** that holds
*every* equivalent form of the program at once, grows it by applying rewrite
rules to saturation, and then **searches** for the fastest concrete program
embedded in it by *measuring real execution time*, not by a cost model.

The compiler has exactly one place where rewriting and pattern-matching happen:
**egglog** (an e-graph + Datalog engine). This is a hard architectural law —
`AGENTS.md:47-48`, `spec.md:106`: *"All graph pattern matching and op selection
must be expressed in egglog rewrites. Do not add Rust-side LLIR graph
post-passes."* Backends contribute rewrite rules; they never post-process the
extracted graph.

### The pipeline

```
Frontend (GraphTensor / PyTorch export)
  │   records
  ▼
LogicalGraph (SSA petgraph)                         src/logical/ir.rs
  │   render() → egglog program text (HLIR)
  ▼
Assembled egglog program  = preamble + per-op snippets + backend matchers
  │                                                 src/egglog_core/egglog_snippet.rs
  ▼
egglog EGraph  ──saturate(multi-stratum schedule)──►  saturated e-graph
  │   (prop · main_ruleset · subst-walk · backend · fixpoint-invariants · …)
  ▼
Compact serialized snapshot                          crate luminal_snapshot
  │
  ▼
SEARCH  (runtime-owned)                              crates/*/src/search.rs
  │   producer_index + SamplingSpace → genome (GA)
  │   per genome:  extract → dps_rewrite → decode_layout → bufferize → PROFILE
  ▼
Best BufferIrGraph  →  runtime load + execute
```

### The five big ideas

1. **One IR, many forms.** The program lives as a set of e-classes. "HLIR" and
   "LLIR" are not two data structures — they are two *vocabularies of
   constructors inside the same e-graph*. Lowering is a `union` between an HLIR
   term and an LLIR term (§I.6, §II.8).

2. **The shapetracker is symbolic math in the e-graph.** There is no
   `ShapeTracker` struct of strides. A layout is an **affine index map**
   $c \mapsto \text{offset}(c)$ built from symbolic integer expressions
   (`IntExpr`) over coordinate variables. Reshapes/transposes/slices/broadcasts
   are **index-map composition**, carried out by a substitution primitive, and
   equivalent layouts collapse into one e-class by congruence (§I.4, §I.5).

3. **Correctness lives in the rules, performance lives in the search.** Every
   selectable program is *semantically equivalent by construction* — a backend
   match rule's premises *are* its correctness proof (`src/layout/ir.rs:550-558`).
   The search therefore never checks outputs; it only *times* candidates
   (`spec.md:139`).

4. **"Describes vs decides."** Core (`crate luminal`) builds and saturates the
   e-graph, draws candidate programs ("genomes"), extracts them, and plans
   memory. It *decides nothing*. Each runtime crate owns the loop that *prices*
   genomes and keeps the best (`src/lib.rs:26-54`). Pricing = real profiling.

5. **Search is evolutionary over e-graph choices.** A "genome" is a choice of
   one producer e-node per e-class. Search is an elitist `(1+λ)` evolutionary
   strategy: mutate the best genome, extract, bufferize, run it, keep the
   fastest (§I.8, §II.11).

---

# Part I — The Theory

This part is self-contained mathematics. No code. Part II maps every object here
onto source.

## I.1 Equality saturation and e-graphs

An **e-graph** is a data structure representing a congruence relation over terms.
It consists of:

- **E-nodes**: a function symbol applied to child **e-classes**,
  $f(c_1,\dots,c_k)$.
- **E-classes**: sets of e-nodes proven equal. Write $[\,t\,]$ for the class of
  term $t$.

The e-graph is closed under **congruence**: if $c_i = c_i'$ for all $i$ then
$f(c_1,\dots,c_k) = f(c_1',\dots,c_k')$. Equivalently, e-nodes are
**hash-consed**: two structurally identical e-nodes (same symbol, same child
classes) are the same e-node, so merging two classes can cascade into further
merges (this is *congruence closure*).

A **rewrite** $\ell \to r$ is applied non-destructively: the engine finds all
matches of the pattern $\ell$, instantiates $r$, and asserts
$[\ell] = [r]$ by **unioning** the two classes. The original form is never
deleted. Running all rewrites until no rewrite adds a new e-node or a new
equality is **equality saturation**; the result is a compact representation of a
possibly-exponential set of equivalent programs.

**egglog** augments this with Datalog. Alongside constructor e-nodes it supports:

- **Relations** $R(x_1,\dots)$ (facts), derived by rules, used to *gate*
  rewrites (a rewrite fires only when a supporting fact has been derived).
- **Functions** $g(\cdot)$ with a **merge** policy: when two values are asserted
  for the same key, `merge` combines them (e.g. keep the max), and
  `:no-merge` makes a disagreement a hard error.
- **Rulesets** and **schedules**: rules are grouped; a schedule says which groups
  run, in what order, and to what fixpoint (`saturate`, `repeat n`, `seq`).

Two egglog idioms are load-bearing in Luminal and worth stating abstractly:

- **Demand/supply (guarded, bottom-up computation).** To compute a recursive
  function $F$ over a term without non-termination, a *demand* relation
  $\mathrm{demand}_F(t)$ is seeded at the consumer; rules propagate demand to
  subterms; and a *state/output* relation supplies $F(t)$ once its children's
  outputs exist. This is how strides, index composition, and layout discovery
  are computed (§I.4, §I.5).

- **`:no-merge` as a runtime invariant checker.** A function declared `:no-merge`
  asserts "this key has exactly one value." Deriving the value from two
  independent sources and letting a disagreement *panic* turns a soundness
  invariant (e.g. "a value has one dtype") into an automatically-checked
  assertion.

### Monotonicity and soundness

Equality saturation is sound only if every asserted equality is **true for all
admissible inputs**. Because classes only ever *merge* (never split), any
*conditional* rewrite must fire on a premise that, once true, stays true —
otherwise a later merge could retroactively invalidate an earlier,
already-materialized equality. Luminal calls this the *monotone-stable* or
*shrink-stable* requirement, and it dictates the shape of the bounds lattice
(§I.2) and the substitution well-definedness guard (§I.5).

## I.2 The symbolic integer algebra

All shapes, strides, offsets, and indices are elements of a single inductively
defined algebra $\mathcal{E}$ of **integer expressions**. Its generators:

$$
\begin{aligned}
e ::=\;& n                      &&\text{(integer literal, } n\in\mathbb{Z})\\
     |\;& v                      &&\text{(symbolic unknown / dynamic dim, a variable)}\\
     |\;& \chi_{s,k}             &&\text{(coordinate variable: axis } k \text{ of shape } s)\\
     |\;& e_1 + e_2 \mid e_1 \cdot e_2 \\
     |\;& e_1 \,\underline{/}\, e_2 \mid e_1 \bmod e_2 \mid \lceil e_1/e_2\rceil &&\text{(trunc-div, trunc-rem, ceil-div)}\\
     |\;& \min(e_1,e_2) \mid \max(e_1,e_2)\\
     |\;& [\![\,b\,]\!]           &&\text{(0/1 indicator of a boolean } b)
\end{aligned}
$$

Booleans $b$ are a *separate sort* with a single primitive comparison
$b ::= (e_1 < e_2)$; all other comparisons are discrete-integer spellings of it
($a\le b \equiv a < b{+}1$, $a > b \equiv b < a$, etc.).

Over $\mathcal{E}$ the e-graph maintains three layers of theory.

### (a) Commutative ring + lattice structure

Addition and multiplication form a commutative ring; $\min,\max$ form a lattice;
division is a **partial** operator family (trunc toward zero with the remainder
carrying the **dividend's** sign — C/Rust/LLVM `sdiv`/`srem`; ceil toward
$+\infty$). The e-graph is taught the laws as rewrites: additive/multiplicative
identities ($e+0=e$, $e\cdot 1 = e$, `:subsume`d), constant folding,
commutativity, associativity, and **like-term collection**
($x\cdot a + x\cdot b = x\cdot(a+b)$).

**The fence doctrine.** The detonation hazard is **distributivity
specifically**, not AC. Commutativity and associativity only *permute* existing
structure, so hash-consing dedupes their output and no new terms accrue;
distributivity *manufactures* fresh $\mathrm{IntMul}$ nodes per firing, and with
the **zero e-class** (endemic to an indexing IR — fold-base accumulators, zero
map entries, $\mathrm{IntMul}(0,c)$ images) present it ignites a non-terminating
multiplication pump that doubles the constant pool per round until `i64`
overflow (verified: `egglog_preamble.egg:1483-1523`, *"the zero e-class is the
engine; distributivity is the only ring rule that ignites it"*). Hence
commutativity/associativity run freely; the **like-term combination** rules fire
only under coefficient-bound **certificates** (see (c); e.g. both coefficients'
lower bounds positive, `:3263-3396`); and raw distributivity is confined to a
bounded/backward-only ("factoring") stroke rather than a free saturating
rewrite. The general principle — *a ring rule that mints new scalar terms must be
gated so it cannot feed itself* — is why every division/stride rule below carries
a certificate rather than a point-in-time test.

### (b) Interval abstract interpretation (the bounds lattice)

Each class $c$ carries an optional interval $[\,\underline{c},\,\overline{c}\,]
\subseteq \mathbb{Z}\cup\{\pm\infty\}$ via two functions:

$$
\mathrm{lb}:\mathcal{E}\to\mathbb{Z},\quad \mathrm{ub}:\mathcal{E}\to\mathbb{Z}
$$

with **monotone merge**: new lower bounds are combined by $\max$, upper bounds by
$\min$. So bounds only ever **tighten** — the lattice moves in one direction,
which is exactly the monotone-stability §I.1 requires. Bounds are seeded on
dynamic dims from the program's declared symbol ranges and propagated through the
ring by interval arithmetic (e.g. $\mathrm{lb}(a+b)=\mathrm{lb}(a)+\mathrm{lb}(b)$).

A parallel pair of functions does **abstract interpretation of tensor values**
(not just index math): each logical tensor carries a value-interval, propagated
through the ops (e.g. the interval of $a+b$ is the interval sum). This is what
lets integer ops be lowered **proof-gated**: an integer add is only lowered to a
non-wrapping kernel where the value-interval proves the result fits in the dtype
width (§I.6, §II.8).

### (c) Divisibility / nonzero certificates

Partiality and degeneracy are handled by monotone **certificate relations** —
facts that, once derived, license a rewrite:

- *nonzero*: a class is certified "provably cannot equal zero" when its interval
  is entirely positive, entirely negative, or it is a coordinate over an extent
  $\ge 2$, or a product of nonzero factors. Division/stride rules require this
  certificate on the divisor/stride rather than a point-in-time "$\ne 0$" check.
- *exact division*: a partial primitive yields $a/b$ only when $b \mid a$; a
  rewrite that cancels simply does not fire otherwise.
- *degenerate axis collapse*: a coordinate over an extent of exactly $1$ is
  unioned with $0$ (a one-point axis never steps). This is why layouts that
  differ only on singleton axes become one class (relevant to CuTe, §III).

**Why it matters for the fork.** This algebra *is* Luminal's "shapetracker."
CuTe's layout algebra (shape:stride tuples with hierarchical nesting) is a
different presentation of the same affine-index-map idea, and it must be mapped
onto (or substituted for) $\mathcal{E}$ and the layout ladder (§I.4, §III).

## I.3 Symbolic shapes

A **shape** is a list of extents, each an element of $\mathcal{E}$:
$s = [d_{r-1}, \dots, d_1, d_0]$, rank $r$. Extents may be literals or dynamic
variables, so a shape like $[\text{batch},\, p+c,\, 4096]$ is a first-class
object; its concrete value is only known at run time through a **valuation**
(a `dyn_map`: variable $\mapsto$ integer).

**Axis convention (critical).** Axes are indexed as **de Bruijn levels from the
innermost (last) dimension**: axis $0$ is the last/contiguous dimension. A
coordinate variable $\chi_{s,k}$ denotes "the index along axis $k$-from-end of a
tensor of shape $s$," and its range is $[0, d_k)$ where $d_k$ is read from $s$
itself. Carrying the *owning shape* in the coordinate gives layouts
**extent-true identity**: two layouts with identical stride formulas over
*different* shapes are *not* the same function and must not be unioned.

## I.4 Layouts as affine index maps, and the refinement ladder

A **layout** is a *physical interpretation map*: it says, for each logical
coordinate $c = (c_{r-1},\dots,c_0)$ of a tensor, which storage **element** (or
**bit**) that coordinate reads/writes. Crucially, a layout is **placement only**
— it carries a domain (shape) and an access **width** in bits, but *no dtype*
(two dtypes of equal width share layouts because they are the same byte map).

Formally a layout is a pair $(\,s,\ \varphi\,)$ with domain shape $s$ and an
offset function $\varphi : \prod_k [0,d_k) \to \mathbb{Z}_{\ge 0}$ (element
index), together with a width $w$. The canonical class of layouts here is
**affine**:

$$
\varphi(c) \;=\; \sum_{k=0}^{r-1} \chi_{s,k}\cdot \sigma_k
$$

for per-axis **strides** $\sigma_k \in \mathcal{E}$. (Plus optionally a constant
origin and a sub-element bit offset for packed sub-byte layouts.)

Luminal represents the *same affine function* at several levels of a
**refinement ladder**, from most specific to most general:

$$
\underbrace{\text{RightMajorContiguous}}_{\sigma = \text{row-major}}
\;\sqsubseteq\;
\underbrace{\text{Strided}(\text{affine chain})}_{\text{explicit } \sigma_k}
\;\sqsubseteq\;
\underbrace{\text{ElementOffsetExpr}}_{\varphi \text{ as one } \mathcal{E}\text{ term}}
\;\sqsubseteq\;
\underbrace{\text{BitOffsetExpr}}_{\text{offset in bits}}
$$

and dually `LeftMajorContiguous` (column-major) sits beside `RightMajorContiguous`.
The two **folds** that relate them:

- **Stride fold.** Row-major strides of shape $[d_{r-1},\dots,d_0]$ are
  $\sigma_k = \prod_{j<k} d_j$ (the running product of inner extents); column-major
  are $\sigma_k = \prod_{j>k} d_j$. These are computed by a demand/supply fold
  over the extent list.
- **Index fold.** Given shape + strides, the offset
  $\varphi(c)=\sum_k \chi_{s,k}\sigma_k$ is built by a right fold. **Zero-stride
  axes contribute nothing** (a broadcast axis), which both avoids a non-termination
  pump and makes a broadcast layout's offset equal to its index-map-composed form
  (§I.5).

**Discovery is bidirectional and automatic.** Upward: a contiguous layout emits
its stride list, which emits its affine chain, which *is* the `Strided` form
(union). Downward: a bit-offset whose expression is divisible by the width
becomes an element-offset; a zero-origin element-offset whose expression matches
a stride fold becomes `Strided`; a `Strided` form whose chain matches a row-major
(or col-major) fold becomes `RightMajor` (or `LeftMajor`). The net effect: **all
spellings of one affine function live in one e-class**, and a backend may match
whichever spelling it can lower, while the search is free to pick the storage
order that profiles fastest.

**Injectivity.** A **(tensor, layout) pair** is **injective** when $\varphi$ is
one-to-one over *that tensor's* domain (no two coordinates alias one element).
The property is keyed on the pair — a `LayoutTensor` — not the bare layout,
because a layout class is shape-erased (its offset expression depends only on
strides and rank), so injectivity is a property of the offset function
*restricted to the tensor's own shape* (verified: the INJECTIVE-WRITE DOCTRINE,
`egglog_preamble.egg:4750-4771`). Injectivity is a **monotone proof token**: set
only where proven, absent = unproven. A *write* is legal only onto an injective
pair ("two coordinates, one offset, last-writer-wins garbage" otherwise); a
*read* through a non-injective layout (a broadcast, a stride-0 view) stays
perfectly legal. This gates the lowering of every op's **output** (§I.6, §II.8).

### The ElementStride layout (the unification workhorse)

Beyond the five ladder constructors there is a sixth, more liberal form:
`ElementStride`, a shape plus a per-axis stride where each stride is either an
**exact coordinate-free expression** or **`Any`** — a don't-care permitted only on
a statically-proven singleton axis. `Any` unifies with any stride, so layouts
that agree except on size-1 axes become one class. This is the closest analogue
in the current system to CuTe's algebra and the most natural target for a CuTe
mapping (§III).

## I.5 Views and index-map composition (substitution)

A **view** (reshape, permute, broadcast, slice, split/merge dims, repeat) is a
reindexing. It is represented as an **index map**

$$
\mu : \text{(output coordinates)} \longrightarrow \text{(input logical index)}
$$

carried as a list of $\mathcal{E}$-expressions — one entry per **input** axis,
each a function of the **output** coordinates — together with a **source-shape
tag** equal to the *input* (parent) shape. (Representation: `IndexMapLit(entries,
source_shape)`; the tag is the parent's shape, stamped at the one recorder mint
site, `src/logical/ir.rs:783-793`; the entries use the output shape's
coordinates, line 796.) Writing a view as $y = \mu(x)$: coordinate $c$ of $y$
reads $x$ at logical index $\mu(c)$. The source-shape tag names *which*
coordinate variables a later substitution will replace (the parent's), which is
why it is part of the map's identity, not mere metadata.

**Composition is substitution.** Composing an outer view $\mu$ with an inner view
$\nu$ means replacing each coordinate variable in $\nu$'s entries by the
corresponding entry of $\mu$:

$$
(\nu \circ \mu)[i] \;=\; \nu[i]\big[\,\chi_{s,k} := \mu[k]\,\big]_k
$$

This is the operator $\sigma \mapsto e[\sigma]$, written `int-subst-of` in the
engine. Likewise, composing a **layout** with a view —
$(\varphi \circ \mu)(c) = \varphi(\mu(c))$ — is the *same* substitution applied
to the layout's offset expression. So "apply a view to a tensor that already has
a strided layout" is literally: substitute the view's index map into the layout's
offset expression, then let the ladder (§I.4) rediscover a constructor form. This
is stated verbatim in the engine — *"A view's layout IS its parent's offset
function pre-composed with the index map:
$\mathrm{offset\_view(out)} = \mathrm{offset\_parent(map(out))}$"* — and
implemented as `composed = (int-subst-of parent_offset map)` over the view's own
output shape (`egglog_preamble.egg:4391-4437`), with a native affine-chain fast
path for selection/zero map entries (`:4440+`).

**Well-definedness (the soundness crux).** Substitution of a *class* (not a
single term) is sound only if it is **universal instantiation**: an equation
true for all coordinates in a box stays true after substituting each variable by
a value *within that variable's own range*. Hence every coordinate substitution
carries a guard: the replacement entry must lie in $[0, d_k)$, checked via the
interval bounds (§I.2) in **smallest-box form** (against the extent's *lower*
bound, so the guard is monotone and never un-fires). An out-of-range substitution
would manufacture false equalities; the guard is the *definition* of when
composition is allowed, not an optimization.

**The substitution walk and closedness.** Substituting a compound expression
requires copying its whole subterm structure under one simultaneous assignment.
This is done by a dedicated **walk primitive** that copies the reachable region
of a term under the assignment $\sigma$ in one shot (simultaneous — so a
transpose, which swaps two axes, composes correctly where naive sequential
rewriting would not). The walk is *gated* by a **closedness lattice**: a class is
"closed under $\mu$" only when every leaf beneath it is admissible (a literal, a
coordinate-free unknown, or an in-range coordinate). Non-closed compositions
**stall fail-closed** (they simply do not compose) rather than producing a wrong
answer.

## I.6 HLIR, LLIR, and lowering as nondestructive rewriting

The program is expressed in two constructor vocabularies *within the one
e-graph*:

- **HLIR (logical).** A `LogicalTensor` is a dataflow value with a shape and
  dtype. The **primitive logical ops** are a small, closed set: elementwise
  unary (`exp2`, `log2`, `sin`, `recip`, `sqrt`, casts, …), elementwise binary
  (`add`, `mul`, `mod`, `less_than`, …), reductions (`sum`, `max`), indexing
  (`gather`, `scatter`, `select`), data generators (`constant`, `iota`), and the
  view operator (`index_map_apply`). An HLIR graph is a *complete bit-level
  logical specification* — the single source of functional truth
  (`spec.md:65`, `AGENTS.md:37`). Higher-level ops (matmul, softmax, transpose)
  are **not primitive**: they are *named spellings* of primitive chains
  (§II.7).

- **LLIR (layout/implementation).** A `LayoutTensor` is a logical value *paired
  with a chosen layout*. A `LayoutTensorOp` is a concrete backend
  implementation: a list of input `LayoutTensor`s and a list of output
  `LayoutTensor`s. These are the nodes that will become kernels.

**Lowering is a union.** A backend provides **match rules** that recognize an
HLIR pattern (plus layout/injectivity/dtype/bounds side-conditions) and
**union** a specialized `LayoutTensorOp` e-node into the e-graph, equal to the
generic `(inputs, outputs)` op. Because the match rule's premises discharge every
applicability condition *before* the op is minted, **presence of the op in the
e-graph is a certificate that it is a valid implementation** — there is no
separate Rust-side legality check and no post-pass. Multiple backends (or
multiple layouts) produce multiple co-existing LLIR implementations of the same
logical value; the search picks among them.

This is the precise sense in which "HLIR → LLIR" is not a translation pass but an
*accumulation of equalities*: after saturation the e-graph contains the logical
program **and** all its legal physical realizations, all equal.

## I.7 Extraction as constrained selection — the genome

After saturation the e-graph is a compact set of programs. To get *one* runnable
program we must **extract**: choose, for each e-class we need, exactly one e-node
that produces it, such that the chosen sub-DAG is acyclic and reaches the program
outputs.

Formally, let $\mathcal{C}$ be the set of `LayoutTensor` classes that have
implementation producers. A **genome** is a partial function

$$
G : \mathcal{C} \rightharpoonup \text{(producer e-node, output slot)}
$$

assigning to each demanded class the concrete implementation e-node that will
realize it. Validity conditions:

1. **Totality over demand.** Every class the extraction walk reaches from the
   output boundary must have a $G$-entry (classes never reached are "dead rows,"
   free to leave unset).
2. **Acyclicity.** The induced *chosen-edge graph* — class $C \to D$ whenever
   $G(C)$'s chosen producer consumes $D$ — must be acyclic.

Extraction then instantiates $G$: starting at the output bindings, it walks down,
emitting the chosen implementation for each class, to produce an **ExtractedGraph**
— a single-valued dataflow DAG of concrete op instances.

The classical e-graph-extraction problem (pick the min-cost acyclic selection) is
NP-hard in general and usually solved by ILP or greedy heuristics with a *static
cost model*. Luminal does **not** use a cost model. It instead *samples* genomes
and *measures* them (§I.8). The only structural subtlety is keeping samples
acyclic:

- The candidate graph's **strongly-connected components** are exactly where a
  choice cycle can form (two classes that can each be described in terms of the
  other — e.g. a layout-copy weld, or $x \equiv T^\top(x)$). A candidate whose
  inputs all leave its component **progresses** and can never close a cycle (the
  condensation of an SCC decomposition is a DAG).
- Sampling therefore proceeds **component-aware**: freely outside components;
  inside a component it elects progressing producers first and inserts
  "re-describing" ones as plumbing, sampling a spanning forest so no cycle closes
  (§II.10).

## I.8 Evolutionary search with measured fitness

Search minimizes measured wall-clock time $T(G)$ of the extracted-and-executed
program for genome $G$ over the valid genome space, using an elitist
$(1+\lambda)$ evolutionary strategy:

$$
\begin{aligned}
&\text{Gen 0: sample } \lambda \text{ random valid genomes } \{G_i\}.\\
&\text{Each gen } t{>}0:\ \text{from incumbent best } G^\star,\
   \text{produce } \lambda \text{ mutants } G' = \mathrm{mutate}(G^\star).\\
&\text{Evaluate each: } T(G) = \mathrm{profile}(\mathrm{bufferize}(\mathrm{dps}(\mathrm{extract}(G)))).\\
&G^\star \leftarrow \arg\min T.
\end{aligned}
$$

Key properties:

- **Mutation only, no crossover.** A mutation re-points a bounded number of
  class choices, admitting a flip only if it keeps the chosen-edge graph acyclic.
- **Fitness is real execution.** There is no analytic cost model; each genome is
  extracted into an actual plan, its memory is planned, and it is *run* on the
  target runtime, timed over a few trials. Invalid genomes (cycle / contract
  breach / un-bufferizable) are simply discarded — the next generation's fresh
  mutations are the repair.
- **Equivalence is assumed, not checked.** Because every selectable LLIR graph is
  semantically equivalent by construction (§I.6), the search *only* ranks speed
  and never validates outputs (`spec.md:139,150`).
- **Dedup + early-stop.** Distinct genomes often extract to the same plan; a
  plan **fingerprint** caches measurements. A candidate whose running mean
  already exceeds the incumbent is abandoned early.

## I.9 Destination-passing and bufferization (memory planning)

The extracted graph is pure value-SSA: every op produces fresh logical values
with no storage. Two post-extraction transforms turn it into a storage plan.

**Destination-passing (DPS).** Each op that *can* write in place is rewritten to
its **destination-passing form**: it gains one trailing **write-only destination
operand per result**, tied to that result (MLIR's DPS/tie invariant). Initially
each destination is a fresh **poison** (undefined-contents) value carrying the
tied result's shape, dtype, and *layout class*. This exposes, uniformly, every
opportunity for in-place computation and buffer reuse without yet committing to
any.

**One-shot bufferization.** A conflict-analysis pass assigns a **buffer id** to
every value, deciding which destination-ties may actually share storage. For each
tie it admits or rejects in-place reuse using:

- **Writability** — never write a buffer whose alias set includes a read-only
  binding.
- **Read-after-write** — a read of any value aliasing the candidate must
  *provably* precede the overwrite, where "precede" is **DAG reachability**
  (unordered ops are conservative conflicts — the analysis reasons about *all*
  valid schedules, like MLIR dominance).
- **Same-use exemption** — an operand's own read never conflicts with its own
  write (the accumulator / read-modify-write case).
- **Repair on rejection** — a rejected tie gets fresh storage, preceded by a copy
  iff the result's pre-write contents must equal the operand's. Every tie is
  repairable, so rejection is never an error.

Boundary values pinned to caller buffers are pre-unioned in an alias union-find.
The output is a **BufferIrGraph**: the final list of storage events (allocate,
compute-into, free) that the runtime executes.

---

# Part II — The Implementation

Maps Part I onto source. Paths are relative to `refs/luminal/`.

## II.0 Module map (`src/lib.rs`)

`src/lib.rs:1-96` declares the core modules. The orienting comment
(`src/lib.rs:26-54`) states the **"describes vs decides"** split verbatim: core
keeps *"the logical program, the egglog assembly, `dps_rewrite`,
`layouts::decode_layout_table`, `bufferize`, the IR types"* plus the two
choice-free halves of search (`extraction`, `search_support`); the
implementation-pricing loop lives in the runtime crates.

| Concern | Module(s) |
|---|---|
| E-graph vocabulary (the preamble) | `src/egglog_core/egglog_preamble.egg` |
| Program assembly + egraph constructor | `src/egglog_core/egglog_snippet.rs` |
| Substitution walk primitive | `src/egglog_core/subst_primitive.rs` |
| Logical op registry + per-op `.egg` | `src/logical/op/*`, `src/logical/ir.rs` |
| Logical helpers (matmul/transpose/broadcast) | `src/logical/helper/*` |
| Symbolic integers (Rust side) | `src/shape/expression.rs`, `src/shape/symbol.rs`, `src/shape/bounds.rs` |
| Layout sort + decoding | `src/layout/layouts.rs`, `src/layout/element_stride.rs` |
| LLIR op traits + extraction site | `src/layout/ir.rs` |
| Reading IntExpr back from e-graph | `src/index_expr.rs` |
| Extraction + genome + sampling space | `src/extraction.rs`, `src/extraction/viability.rs` |
| Genome sampling + search reporting | `src/search_support.rs` |
| DPS rewrite | `src/layout/dps.rs` |
| Bufferization | `src/buffer/bufferize.rs`, `src/buffer/buffer_tensor_ir.rs`, `src/buffer/arena.rs` |
| Compact serialized e-graph | crate `luminal_snapshot` (`src/snapshot/mod.rs` re-exports) |
| Frontend GraphTensor API | `src/graph.rs`, `src/frontend/*` |
| A runtime (search loop + kernels) | `crates/luminal_reference`, `crates/luminal_cuda_lite` |

## II.1 The e-graph vocabulary and program assembly

### The sorts (`egglog_preamble.egg:17-410`)

The single `(datatype* …)` block declares every core sort (§I maps directly):

- `IntExpr` (`:20-91`): `IntLit`(`:21`), `IntVar`(`:29`), `CoordVar Shape i64`
  (`:45` — the owned coordinate $\chi_{s,k}$, §I.3), `IntAdd`/`IntMul`(`:46-47`),
  the division family `IntTruncDiv`/`IntTruncRem`/`IntCeilDiv`(`:67-69`), the
  lattice pair `IntMin`/`IntMax`(`:80-81`), and the bool bridge
  `IntCastFromBool`(`:90`).
- `BoolExpr` (`:98-105`): `BoolLit`, the single comparison `BoolLessThanInt`.
- `IntAffineExprList` (`:138-140`): the affine chain of §I.4, one summand
  $\chi_{s,k}\cdot\sigma_k$ per axis.
- `IntExprList` (`:142-145`), `Shape = ShapeLit IntExprList` (`:153-155`).
- `LogicalTensor` (`:173-176`): `LogicalTensorInputLit LogicalId Shape Dtype`
  (a declared input) and `LogicalTensorNamed LogicalId` (a name annotation,
  unionable with any value). **Note: per-op logical constructors are not here —
  they are spliced in** at `;; @SPLICE logical-constructors` (`:412`).
- `IndexMap = IndexMapLit IntExprList Shape` (`:189-191`) — the view map $\mu$ of
  §I.5 (entries + source shape).
- `BitWidth` (`:198-200`); `Layout` (`:203-242`) — the five ladder constructors:
  `BitOffsetExpressionLayoutLit`(`:236`), `ElementOffsetExpressionLayoutLit`
  (`:237`), `StridedElementLayoutLit Shape IntAffineExprList BitWidth`(`:239`),
  `RightMajorContiguousElementLayoutLit`(`:240`),
  `LeftMajorContiguousElementLayoutLit`(`:241`).
- `LayoutTensor = LayoutTensorLit LogicalTensor Layout` (`:266-268`) — the LLIR
  value of §I.6.
- `LayoutTensorOp = LayoutTensorOpLit LayoutTensorList LayoutTensorList`
  (`:273-282`) — the generic implementation (input list, output list).
- Buffer level: `BufferTensor`, `BufferId`, `Access`(ReadOnly/ReadWrite),
  `FreedBy`(CallerFrees/ProgramFrees), `BufferInput`, `BufferOutput`
  (`:292-354`).
- `Dtype` (`:370-408`): the full dtype set.

Invariant-enforcing functions follow (`:422-468`): `rank-of`, `shape-of`,
`dtype-of`(`:451`, `:no-merge` — §I.1 invariant checker), `bits-of`,
`layout-of`, etc. The design note at `:425-450` explains why `dtype-of` is
`:no-merge` and declared/concluded per-op rather than by a generic
first-input rule.

### Assembly (`egglog_core/egglog_snippet.rs`)

The full program is **spliced together** from the preamble plus per-op snippets.
There are **10 splice anchors** (`SpliceCategory`, `:24-77`):
`LogicalConstructors`, `LayoutOpConstructors`, `Dtype`, `Seed`, `Shape`,
`Rewrites`, `Forward`, `Match`, `Coordinate`, `Fixpoint`. `assemble(core,
snippets)` (`:91-113`) replaces each `;; @SPLICE <cat>` marker with that
category's snippets in registration order. `assembled_program_for(matchers)`
(`:193-207`) concatenates: the preamble + `element_stride` snippets +
`built_in_logical_ops()` snippets + `built_in_logical_helpers()` snippets + the
runtime's `matchers` snippets.

`new_egraph()` (`:135-185`) is the **only sanctioned e-graph constructor**. It:
registers the ElementStride primitives (`:139`); sets `no_decomp = true`
(tree-decomposition off — a measured perf regression) (`:143`); registers
`bigint-to-i64` and the partial `exact-div` (i64 + BigInt) (`:144-162`); and
registers the **substitution walk primitives** `Subst` and `MemoizedSubst`
(`:168-183`) — the `unstable-subst` vendored primitive with skip-set
`{int-subst-of}` and cache table `int-subst-cache`.

> **Fork note.** This splice architecture is exactly how a CuTe backend plugs
> in: add `LayoutOpConstructors` + `Match` snippets for CuTe ops, and (if CuTe
> layouts become first-class) new `Layout`-sort constructors with their discovery
> rules. No core file needs editing beyond the preamble's new sort constructors.

## II.2 The saturation schedule and rulesets

Rulesets declared in the preamble (verified counts at clone time):
`main_ruleset` (`:476`, ~261 rules — the ring, layout ladder, IndexMap, and
value-bounds rewrites), `prop` (`:941` — only `shape-of`/`dtype-of`/`rank-of`
propagation; a test enforces this exactly, `egglog_snippet.rs:369-404`),
`fixpoint-invariants` (`:4931` — the terminal invariant stratum),
`cleanup` (`:566` — deletes HLIR after lowering), `subst-walk` (`:4378` — the
substitution walk), `backend` (`:4868` — **all** backend HLIR→LLIR match rules;
the `;; @SPLICE match` anchor is at `:4870`), plus `layout-tensor-op-metadata`
(`:478`) and `materializing-copy-mint` (`:490`).

The production **schedule** is a Rust constant per runtime. The cuda-lite form
(`crates/luminal_cuda_lite/src/bindings.rs:386`; core default at
`src/extraction.rs:4644`; helper-test form at `src/logical/helper/mod.rs:49`):

```lisp
(run-schedule
  (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk))
  (saturate (run element-stride-discovery))                ; cuda-lite only
  (saturate (saturate (run main_ruleset) (run backend) (run prop)) (run subst-walk))
  (run materializing-copy-mint)
  (run layout-tensor-op-metadata)
  (saturate (run cleanup))                                  ; cuda-lite only
  (saturate (run fixpoint-invariants)))
```

This is the operational heart. Note the stratification (§I.1): (1) seed shapes;
(2) rewrite the logical/layout algebra *without* backend matchers (let the layout
tables settle) with the substitution walk as an **outer stratum** so the ring
saturates between walk generations (`egglog_preamble.egg:4142-4156`); (3) run
backend matchers to mint LLIR; (4) mint materializing copies and op metadata;
(5) delete HLIR; (6) run the terminal invariant stratum, whose optimistic
`:no-merge` rules are only legal *because* the main run has already saturated
(`egglog_preamble.egg:4911-4931`).

A single seed rule gives every value a default layout:
*"right-major contiguous `LayoutTensor` for every logical"*
(`egglog_preamble.egg:4844-4855`), from which the ladder discovers all other
spellings.

## II.3 The IntExpr machinery (ring, lattice, bounds)

Ring laws (all `:ruleset main_ruleset`, `egglog_preamble.egg`): additive &
multiplicative identities with `:subsume` (`:1347-1352`); constant folding for
`IntAdd`/`IntMul`/division/`IntMin`/`IntMax` (`:1355-1442`); commutativity
(`:1452-1471`); like-term collection $x a + x b = x(a+b)$ gated on coefficient
bounds (`:3380-3397`).

Bounds lattice (§I.2b): `lower-bound-of (IntExpr) BigInt :merge (max old new)`
and `upper-bound-of … :merge (min old new)` (`:1550-1551`) — monotone tightening.
Value-level analogues `value-lower-bound-of` / `value-upper-bound-of` on
`LogicalTensor` (`:1845-1846`) drive proof-gated integer lowering (§II.8).

Certificates (§I.2c): the `provably-cannot-union-with-zero` family
(`:2053-2300`) — positive-lower, negative-upper, coordinate-of-extent-$\ge2$,
product-of-nonzeros; degenerate-axis collapse "`CoordVar` of extent exactly 1
collapses to zero" (`:2132`); "a `CoordVar` divided by its own extent is zero"
(`:2157`). The partial `exact-div` primitive is bound in `new_egraph`
(`egglog_snippet.rs:153-162`).

The **read side** — turning a saturated `IntExpr` class back into an evaluable
numeric tree — is `src/index_expr.rs`. `IotaExpr` (`:14-33`) mirrors $\mathcal{E}$
(`Lit`, `Var`, `Coord`, `Add`, `Mul`, `TruncDiv`, `TruncRem`, `CeilDiv`, `Min`,
`Max`, `LessThanCast`); `eval_with_dims` (`:42-67`) evaluates it at output
coordinates + a dyn-map (this is the *execution-time* shapetracker — the kernel
computes each element's address by evaluating this per output coordinate);
`parse_int_expr` (`:78-261`) backtracks over all equal spellings of a class
(they denote one function) with a cycle-tainting memo and an **owner-shape
guard** (a `CoordVar` owned by a foreign shape must not be read as an
out-coordinate, `:177-197`).

## II.4 Symbolic shapes and the strided-indexing fold

Rust-side symbolic integers are `src/shape/expression.rs` (`IntExpr`, the type
the frontend builds shapes and index maps from) and `src/shape/symbol.rs`
(`Symbol`, `DynMap`); dynamic-dim ranges are `src/shape/bounds.rs`
(`SymbolBounds`, `program_dimensions`). The recorder renders these into egglog
`IntExpr` terms (`src/logical/ir.rs:490`, `:1557` show `(IntVar …)` emission).

Inside the e-graph, §I.4's folds are demand/supply machines in
`egglog_preamble.egg`:

- **Right-major strides** (`:3408-3468`): `right-major-strides-demand/-state`
  fold the extent list to strides $[\,d_1 d_0,\ d_0,\ 1\,]$.
- **The construction seam** `strided-lists (Layout Shape IntExprList IntExprList
  BitWidth)` (`:3478`) records "this layout was built from these (dims,
  strides)"; rows are written only at construction sites, never recovered.
- **affine-zip** (`:3482-3516`) zips a seam row into the `IntAffineExprList`
  chain — one `(IntMul (CoordVar sh axis) stride)` summand per axis — and unions
  the layout with `StridedElementLayoutLit shape chain bits`.
- **Left-major strides** (`:3535-3624`) mirror the above for column-major.
- **The index fold** `strided-indexing-foldr` (`:3633-3719`) collapses the chain
  to the offset $\varphi(c)=\sum_k\chi_{s,k}\sigma_k$; **it is the one site that
  mints `CoordVar`s** (`:3697`, de Bruijn level = tail length), and **zero-stride
  axes mint nothing** (`:3702-3719` — the broadcast-axis / non-termination
  argument of §I.4).

Downward discovery: bit-offset → element-offset when width divides
(`:3882-3895`); the rest of the ladder is driven by matching folded forms.

## II.5 The layout ladder and ElementStride

The ladder constructors are the five `Layout` variants (§II.1). The Rust side
that *reads a layout class back* is `src/layout/layouts.rs`: a per-constructor
decoder architecture (the `EgglogConstructor`/`LayoutFacts` traits in
`src/egglog_core/egglog_utils/eclass.rs`, designed in
`docs/design/eclass-decoder.md`) yields a `DecodedLayout` carrying the decoded
spelling set, consumed by bufferize and codegen. `decode_layout_table(view,
graph, who, cache)` decodes every `LayoutTensor`'s layout once per search.

`ElementStride` (`src/layout/element_stride.rs`): the sixth layout form
(§I.4). `ElementStride` is `Any` or `Exact(IntExpr)` (`:27-31`);
`ElementStridedLayout { shape, strides, width }` (`:33-38`); `validate` requires
`Any` only on a static singleton and `Exact` coordinate-free (`:40-65`); its
`LayoutFacts` impl gives `span` and `element_index = Σ coordᵢ·strideᵢ`
(`:104-144`). It registers two custom primitives — `element-stride-compatible`
(Any unifies with anything, `:208-237`) and `element-stride-prefer-any`
(`:239-270`) — via `register` (`:272-275`), and contributes a separate
`element-stride-discovery` ruleset (`:277-292`, the `.egg` files in
`src/layout/`) that composes **linear certificates** (shape-owned expr + BigInt
coefficient list + origin) to publish nonnegative zero-origin layouts.

## II.6 IndexMap substitution and the walk primitive

The substitution operator (§I.5) is `int-subst-of (IntExpr IndexMap) IntExpr`, a
**constructor** (`egglog_preamble.egg:3996`), demanded via `int-subst-demand`
(`:3995`). Leaf arms: a literal maps to itself (`:4005-4009`); an `IntVar` maps
to itself (coordinate-free, `:4014-4018`); a `CoordVar` maps to the map's
entry **guarded in-range** — `entry_lower ≥ 0` and `entry_upper ≤ extent_lower−1`
(`:4019-4058`, the smallest-box well-definedness guard of §I.5), plus a
structural "entry is a same-extent coordinate" arm (`:4067-4082`). `IntMul` with
a zero image collapses to `0` (`:4104-4123`).

Compound images are built by the **substitution walk** (`:4142-4191`): one
uncached call per canonical `(root, σ)` to the vendored `unstable-subst`
primitive (implemented in `src/egglog_core/subst_primitive.rs`, with audited
deltas: subsumed source rows copy born-retired; memo constructors skipped; walk
errors print reasons). It runs from a `:naive` rule in the `subst-walk` ruleset
as the **outer stratum** of the schedule (§II.2) so the ring saturates between
generations; `memoized-int-subst` reuses images via `int-subst-cache (IntExpr
SigmaMap) :merge old` where `SigmaMap = (Map IntExpr IntExpr)` (`:4246-4251`).
The walk is gated by the **closedness lattice** `int-subst-closed` (`:4192-4299`)
— witness-only recursion; non-closed compositions stall fail-closed (§I.5).

Logical-level view composition is the op `LogicalIndexMapApply (LogicalTensor
IndexMap Shape)` (`src/logical/op/index_map_apply/constructor.egg`); adjacent
applies compose by substituting the outer map into the inner map's entries and
unioning to a single composed view (`index_map_apply/compose.egg`), while a
view's output layout is minted contiguous and then re-discovered via the ladder
(`index_map_apply/forward_layout.egg`).

## II.7 Logical op anatomy

`src/logical/op/mod.rs` defines the `LogicalOp` trait (`:51-83`): each op owns its
egglog constructor name, display name, child ports, renderer, and **`snippets()`**
(its `.egg` files tagged with splice categories). `built_in_logical_ops()`
(`:244-300`) is the registry (~48 ops). Each op lives in `src/logical/op/<name>/`
with a canonical file set. Worked example, `add` (`src/logical/op/add/`):

| File | Category | Role |
|---|---|---|
| `constructor.egg` | LogicalConstructors | `(constructor LogicalAdd (LogicalTensor LogicalTensor) LogicalTensor)` |
| `dtype.egg` | Dtype (`:ruleset prop`) | dtype propagation; duplicative `set` as a `:no-merge` tripwire |
| `shape.egg` | Shape (`:ruleset prop`) | shape propagation, gated on operand shapes being one class |
| `forward_layout.egg` | Forward (`main_ruleset`) | mint an output `LayoutTensorLit` per operand layout |
| `rewrite.egg` | Rewrites (`main_ruleset`) | commutativity — **exact integer dtypes only** (float bit-exactness preserved) |
| `rewrite_2.egg` | Rewrites | associativity — Int/Int64 only (f32 add is not associative) |
| `fixpoint.egg`/`fixpoint_2.egg` | Fixpoint (`fixpoint-invariants`) | shape closure forward & backward |
| `value_bounds.egg` | Rewrites | interval arithmetic for value bounds (§I.2b) |

`mod.rs` (`:31-70`) just lists these as `EgglogSnippet{category, text:
include_str!(...)}`. This per-op pattern is how §I.6's HLIR vocabulary and its
algebraic laws are declared.

**Composite ops are helpers, not primitives.** `src/logical/helper/` defines
`LogicalHelperMatrixMultiply`, `LogicalHelperMatrixTranspose`,
`LogicalHelperBroadcastAxis` (`helper/mod.rs:25-41`). A helper is a *named
spelling* of a primitive chain with paired **recognize** and **expand** rules
that union the helper's class with the chain's. For matmul
(`helper/matrix_multiply/`): `recognize.egg` turns
$\mathrm{reduce\_sum}(\mathrm{mul}(\mathrm{bcast}(a),\mathrm{bcast}(b^\top)),0)$
into `LogicalHelperMatrixMultiply(a, transpose(b^\top))`, and `expand.egg` turns
it back — so `out[m,n]=\sum_k a[m,k]b[k,n]` and its primitive chain coexist in
one e-class. A GEMM backend matches the helper; a fusion backend matches the
chain.

> **Fork note.** A CuTe GEMM lowering attaches a `backend`-ruleset match rule to
> `LogicalHelperMatrixMultiply` (or the chain) and mints a CuTe op. This is the
> single most important seam for the CuTe fork (§III).

## II.8 Backend lowering — the Match rules and the op instance

A backend op is defined in one folder. Worked example,
`crates/luminal_reference/src/ops/add/`:

- **`match_functional_constructor.egg`** (LayoutOpConstructors):
  `(constructor LayoutTensorOpAddFunctionalGeneric (LayoutTensor LayoutTensor
  Layout) LayoutTensorOp)` — declares the LLIR op.
- **`match_functional.egg`** (`:ruleset backend`): the match rule. Its premises
  (`match_functional.egg`): both operands are `LayoutTensorLit`, the output is
  `LogicalAdd`, the output layout is `RightMajorContiguousElementLayoutLit`, the
  output layout tensor is `(injectivity-of …) = (Injective)`, and dtype-gated
  — non-integer *ungated*, Int/Int64 **proof-gated** on
  `value-lower/upper-bound-of` fitting the width (§I.2b, §I.6). The action mints
  `(LayoutTensorOpLit inputs outputs)` and **unions** it with
  `(LayoutTensorOpAddFunctionalGeneric in1 in2 out_layout)` (§I.6: lowering is a
  union).

The **validity contract** (`src/layout/ir.rs:550-558`): an enode carrying a
registered constructor is extractable *by construction* — the match rule proved
applicability, so the matcher never declines.

The **Rust instance** side (`crates/luminal_reference/src/ops/add/mod.rs`):
`AddFunctional` (out-of-place) and `AddFunctionalDps` (destination-passing) both
implement `LayoutIrOp = BufferTensorIrOp + Bufferizable + ToDps + CloneLayoutIrOp`
(`src/layout/ir.rs:210-213`). `AddFunctional::to_dps()` → `AddFunctionalDps`,
whose `alias_info` declares operand 2 (`dest0`) **must-alias** result 0 and is
write-only — the DPS contract bufferize reads (§I.9). The **matcher**
`AddFunctionalMatcher` implements `OpMatcher` (`src/layout/ir.rs:565-612`):
`egglog_constructor()`, `metadata_slots()` (which children are metadata, e.g.
`out_layout` at index 2), `snippets()`, and `extract(site) -> Box<dyn
LayoutIrOp>` (infallible). The **kernel** (a plain Rust fn) is dispatched by
`TypeId` from a label→fn table in `crates/luminal_reference/src/kernels.rs`.

The runtime registers its matcher set (`built_in_matchers()`); cuda-lite exposes
it via `matchers()` (`crates/luminal_cuda_lite/src/runtime.rs:305`). The matcher
set is a *parameter* to assembly and extraction — core hardcodes no backend.

## II.9 Serialization — the compact snapshot

After saturation the live egglog `EGraph` is serialized to the **compact
snapshot** in the separate `luminal_snapshot` crate (`src/snapshot/mod.rs`
re-exports `compact`; `src/lib.rs:107` aliases it as `egraph_serialize`). The
compact `EGraph` holds `nodes: DenseMap<NodeId, Node>`, `Node { op, children,
eclass, subsumed, … }`, and precomputed `GraphIndexes` (class → nodes, op →
nodes), read via `SerializedIndex` (`src/layout/ir.rs:266-345`). Extraction and
matchers only ever read this snapshot, never the live egraph.

## II.10 Extraction and the genome

`src/extraction.rs` ("decides nothing," `:1-11`). Core types:

- `ProducerChoice { enode: NodeId, output_index: usize }` (`:625-628`) — one
  concrete producer and which of its output slots carries the class.
- `Genome { choices: HashMap<ClassId, ProducerChoice> }` (`:638-640`) — §I.7's
  $G$. Totality over demanded classes; unset-but-undemanded rows are legal.
- `extract_layout_ir_with_genome_and_matchers(egraph, genome, matchers)`
  (`:650-658`): build an `Extractor`, `apply_viability_filter`, walk from the
  `BufferOutput` bindings instantiating the genome's chosen producer per demanded
  class; unassigned output slots get anonymous waste destinations. Returns an
  `ExtractedGraph { dag: DiGraph<ExtractedNode, ExtractedEdge>, outputs }`
  (`src/layout/ir.rs:627-653`), where `ExtractedNode` is `BufferInput |
  LayoutOp(OpNode) | BufferOutput`.
- `producer_index_with_matchers` (`:665`) — per class, the candidate producers
  (the raw material for sampling).

The acyclicity structure (§I.7) is `SamplingSpace` (`:763-779`):
`candidate_inputs` (per-candidate input classes = the candidate-graph edges),
`intra_sources` (inputs inside the same SCC; empty ⇒ *progressing* candidate),
`components` (the non-trivial SCCs), `component_of`. `from_candidate_inputs`
(`:786-851`) builds the candidate graph, runs **Tarjan SCC**
(`petgraph::algo::tarjan_scc`, `:809`), keeps components of size $>1$ or with a
self-loop, and computes intra-component sources. `chosen_edges` (`:872`) is the
genome's realized dependency graph, the one the sampler keeps acyclic.

## II.11 The evolutionary search loop (runtime)

`crates/luminal_reference/src/search.rs` is the canonical "decides" half
(cuda-lite mirrors it). `CompileOptions` (`:106-139`) defaults:
`generations = 8`, `generation_size = 8`, plus `mutations`, `trials`, `seed`,
memory budgets. `SearchOutcome` (`:155-160`): `best_plan: BufferIrGraph<…>`,
`best_genome`, `best_nanos`, `plans_profiled`.

`search_implementations_with_ops` (`:254-565`) implements §I.8 exactly:

1. **Memory pre-pass** `egraph_postpass::run` (`:263`) prunes oversized tensors /
   producer classes beyond the budget.
2. Build the `ExtractionSession`, `producer_index` (`:298`), `SamplingSpace`
   (`:319`); closures `random_genome = sample_genome(index, space, …)` and
   `mutate = mutate_genome(parent, index, space, classes, …)` (`:321-324`).
3. `best: Option<(nanos, Genome, plan)>`. For each of 8 generations: if no
   incumbent, sample 8 random genomes; else mutate the incumbent 8 times
   (`:356-371`). This is the elitist $(1+\lambda)$ strategy.
4. Per genome (`:373-540`): `extract_with_genome` (`:378`); on failure classify
   (cycle / dead-end) and discard. Compute a `plan_fingerprint` (`:429`) and
   consult a `fingerprint → nanos` **dedup cache** (`:430-434`). On miss:
   `dps::dps_rewrite` (`:447`) → `decode_layout_table` (`:448`) →
   `bufferize` (`:454`) → `profile_on_reference_runtime` (`:475`, the *actual*
   run over `trials`, with an **early-stop** cutoff vs. the incumbent). Keep the
   fastest (`:498-513`).

`profile_on_reference_runtime` (`:189-250`) actually executes the plan and
returns measured nanoseconds; `early_stop_exceeded`
(`src/search_support.rs:344`) abandons a candidate whose running mean already
loses to the incumbent. Genome drawing/mutation live in `src/search_support.rs`:
`sample_genome` (`:420`), `mutate_genome` (`:629`), `sample_genome_correlated`
(`:443`), `flip_closes_cycle` (`:591`), plus the reporting vocabulary
`RefusalBreakdown` (`:232`), `SearchTimings` (`:297`), `SearchProgress` (`:84`).

## II.12 DPS and bufferization

`src/layout/dps.rs` — `dps_rewrite(graph)` (`:29`) out-of-place rewrites every
op with a DPS form, appending one write-only **Poison** destination operand per
result (`src/buffer/buffer_tensor_ir.rs::Poison`), each carrying the tied
result's shape/dtype/**layout class** (`dps.rs:60-120`). Idempotent (DPS forms
answer `to_dps() = None`). This realizes §I.9's DPS step.

`src/buffer/bufferize.rs` — `bufferize::<L: PlanLayout>(dps, table)` (`:1739`)
returns `BufferIrGraph<L>` (`:371`). The module doc (`:1-60`) specifies the §I.9
one-shot conflict engine: bottom-up walk; `Must` alias edges from
`Bufferizable::alias_info`; writability / read-after-write (by **DAG
reachability**) / same-use-exemption / cross-operand `Sharing::May` permit; repair
on rejection; boundary values pre-unioned in the alias union-find. The plan is
generic over the decoded layout `L` — this is the seam where `DecodedLayout`
(and, in a fork, a CuTe layout) is consumed.

## II.13 The recorder and frontend

`src/graph.rs` — `Graph { dyn_map, logical: LogicalGraph }` (`:15-22`).
`Graph::tensor(shape, dtype)` (`:38`) declares an input via
`LogicalGraph::input` and returns a `GraphTensor` (id + dims + dtype).

`src/logical/ir.rs` — the SSA recorder. `LogicalOp` (`:115`) is the recorder's op
enum; `Movement` (`:80-97`) enumerates views (`Permute`, `ExpandDim`,
`RemoveDim`, `SplitDims`, `MergeDims`, `Repeat`, `Shrink`); `MapEntry` (`:24-37`)
is the recorded index-map-entry algebra (`Coord`, `Lit`, `Add`, `Mul`, `Div`,
`Rem`, `Min`, `Max`). `LogicalGraph` is a petgraph; `render(&[roots])` (`:1389`)
emits the HLIR egglog text — one `(let vN (Logical… ))` per value in the output
cone, movements lowered to `IndexMapLit` applications. Construction-time view
chaining is `MapEntry::substitute` (`:47-75`); *recorded* view composition is
egglog's job (`int-subst-of`).

The frontend surface (`src/frontend/`) is the ergonomic API — `GraphTensor`
methods in `unary.rs`, `binary.rs`, `movement.rs`, `matmul.rs`, `reduction.rs`,
etc. Movement methods (`permute`, `expand`, `squeeze`, `split_dims`,
`merge_dims`, `repeat`, and the `ViewChain` builder) record `Movement`s that
lower to `IndexMap`s. This layer is "wrapper" — it only *constructs* HLIR.

## II.14 Dynamism

Dynamic dimensions are symbolic `IntVar`s (§I.3) carried through all expressions.
A **valuation** is a `dyn_map: Symbol → usize` (`src/shape/symbol.rs`), supplied
per execution. Symbol ranges seed the bounds lattice via `SymbolBounds` /
`program_dimensions` (`src/shape/bounds.rs`). **Bucketing** (`src/bucketing.rs`,
`docs/design/application-bucketing.md`) allows separate compiles per dynamic
range bucket, dispatched at runtime by the concrete `dyn_map` — so different size
regimes can select different physical plans. Profiling uses representative
dynamic values, but the compiled plan must stay valid across the whole declared
domain (`spec.md:145-149`).

---

# Part III — Reverse-engineering guide: forking onto a CuTe IR

This part is prescriptive. It assumes the goal is to **replace Luminal's layout
representation and/or LLIR op layer with CuTe** (NVIDIA CUTLASS's layout algebra:
hierarchical `Shape : Stride` tuples with composition, complement, and tiling as
first-class algebra) while keeping the equality-saturation front half.

## III.1 What CuTe is, in Luminal's terms

A CuTe `Layout` is a (possibly nested) `Shape:Stride` pair denoting an affine map
from a logical coordinate to a linear index — *the same object* as Luminal's
affine layout (§I.4): $\varphi(c)=\sum_k c_k\sigma_k$, generalized to
hierarchical modes. CuTe's power is its **algebra**: composition
($A \circ B$), complement, logical/zipped/tiled product and divide — closed
operations that Luminal currently reconstructs piecemeal via the stride folds
(§II.4), `int-subst-of` composition (§II.6), and the discovery ladder (§II.5).

So the mapping is natural:

| Luminal object | CuTe counterpart |
|---|---|
| `IntAffineExprList` (affine chain) | a `Shape:Stride` layout |
| stride folds (`right/left-major-strides`) | `make_layout` / `make_ordered_layout` |
| `int-subst-of` (view ∘ layout) | `composition(layout, tiler)` |
| the refinement ladder (contiguous ⟷ strided ⟷ offset) | different *spellings* of one CuTe layout |
| `ElementStride` `Any` on singleton axes | CuTe's size-1 mode coalescing |
| `injectivity-of = Injective` | CuTe layout with no stride-0 / no aliasing (a "compact" injective layout) |

## III.2 The three seams, and what to preserve vs. replace

Decide the **altitude** of the fork. There are three candidate seams, from least
to most invasive:

**Seam A — CuTe only inside the backend (least invasive, recommended first
step).** Keep the entire core e-graph, layouts, and search unchanged. Add a new
runtime crate (mirror `luminal_cuda_lite`) whose **match rules** (`:ruleset
backend`) recognize HLIR patterns and mint `LayoutTensorOp*` CuTe kernels, and
whose `OpMatcher::extract` builds `LayoutIrOp` instances that *carry a CuTe
layout computed from the decoded Luminal layout*. Here CuTe is a codegen detail:
you consume `DecodedLayout` (§II.5) at `extract`/codegen time and translate its
affine chain into a CuTe `Shape:Stride`. **Nothing in core changes.** This is the
fastest path to a working CuTe backend and validates the translation
`DecodedLayout → CuTe Layout` before committing further.

**Seam B — CuTe as a first-class `Layout` constructor (medium).** Add a sixth/
seventh `Layout` constructor `CuteLayoutLit …` to the preamble datatype block
(`egglog_preamble.egg:203-242`), plus:
- **discovery rules** that union a CuTe layout with the existing ladder when they
  denote the same affine map (so CuTe layouts participate in congruence and the
  search can pick them) — model these on the `strided-lists`→`StridedElement`
  seam (§II.4) and the ElementStride discovery ruleset (§II.5);
- a **decoder** (`EgglogConstructor` + `LayoutFacts` impls in `layouts.rs`,
  registered via `OpMatcher::decoders`/`core_decoders`, with the assembly
  tripwire from `docs/design/eclass-decoder.md`) so the runtime can read a CuTe
  layout class back into a `DecodedLayout`;
- forward-layout mints so ops can *offer* CuTe-laid-out outputs.
This makes CuTe layouts selectable by the search (you can profile row-major vs.
CuTe-tiled storage), but keeps `IntExpr`/`IndexMap`/injectivity as the semantic
substrate.

**Seam C — CuTe algebra *replaces* the affine-chain + fold + subst machinery
(most invasive).** Replace `IntAffineExprList`, the stride folds (§II.4), and
possibly `int-subst-of` composition (§II.6) with CuTe layout operations. This is
a rewrite of the core's layout theory and should only be attempted after B is
working. The risk is that CuTe's algebra must then be expressed *as egglog
rewrites with monotone-stable guards* (§I.1), or executed as **custom egglog
primitives** (like `element-stride-compatible`/`exact-div`/the subst walk — see
`new_egraph`, `egglog_snippet.rs:135-185`, and `element_stride::register`,
`element_stride.rs:272-275`). Implementing CuTe composition as a primitive
(a pure function over serialized layout values) is likely the right technique,
mirroring the `unstable-subst` walk primitive.

**Preserve in all cases** (these are the correctness backbone, §I.1, §I.6):
- the `:no-merge` invariant functions (`dtype-of`, `shape-of`, `injectivity-of`)
  and the monotone bounds lattice;
- the rule that *lowering is a union* and *presence = certificate*
  (`layout_ir.rs:550-558`) — any CuTe op must be minted only by a `backend` rule
  whose premises prove equivalence to the authored HLIR;
- the extraction/genome/`SamplingSpace`/GA loop — it is layout-agnostic and
  prices CuTe plans by *measurement*, which is exactly what you want for a
  tiling-heavy IR;
- DPS + bufferize — `bufferize::<L>` is already generic over the plan layout
  `L: PlanLayout`, so a CuTe `DecodedLayout` drops in (§II.12).

## III.3 A concrete migration order

1. **Build Seam A.** New runtime crate; match rules for a few ops (start with
   elementwise `add`/`mul` and `LogicalHelperMatrixMultiply`); `DecodedLayout →
   CuTe Layout` translator; kernels; wire the `search.rs` loop (copy
   `luminal_reference/src/search.rs` structure). Validate numerics against
   `ReferenceRuntime` (the semantic reference, `spec.md:37`).
2. **Add the GEMM seam.** A `backend` match rule on
   `LogicalHelperMatrixMultiply` (§II.7) minting a CuTe GEMM `LayoutTensorOp`,
   dtype/bounds/injectivity-gated like `reference/add/match_functional.egg`.
   This is where CuTe earns its keep.
3. **Promote to Seam B** if you want the *search* to choose CuTe tiled layouts:
   add the `CuteLayoutLit` constructor, discovery unions, decoder, and
   forward-layout mints. Lean on `docs/design/eclass-decoder.md` for the decoder
   contract and the `element_stride*` files as the worked template for a
   custom-layout discovery ruleset.
4. **Consider Seam C** only if the affine-chain representation becomes the
   bottleneck; implement CuTe composition as an egglog primitive.

## III.4 Invariants you must not break

- **Monotone stability** (§I.1): every conditional CuTe rule must fire on a
  premise that stays true after future merges (use the bounds lattice's
  lower-bound / certificate style, never a point-in-time inequality).
- **Owner-shape / extent-true identity** (§I.3): CuTe layouts over different
  shapes must not be unioned just because their stride tuples match; carry the
  shape (as `CoordVar`s already do).
- **Injectivity before writes** (§I.4): a CuTe layout used as a *write*
  destination must carry an `injectivity-of = Injective` proof — CuTe's
  non-injective (stride-0 / swizzled-aliasing) layouts are read-only broadcasts
  unless proven collision-free.
- **Semantic equivalence of every selectable graph** (`AGENTS.md:44`,
  `spec.md:36`): if two CuTe spellings land in one class, they must denote the
  same function over the *entire* valid dynamic-shape domain, or the search can
  silently pick a wrong plan. Debug false unions; never "select around" them.
- **No Rust post-passes** (`AGENTS.md:47-48`): all CuTe fusion/selection lives in
  `backend` rules + extraction, never in a hand-written pass over the extracted
  graph.

---

## Appendix A — Glossary (term → §, source)

| Term | Meaning | Where |
|---|---|---|
| e-graph / e-class / e-node | congruence-closed term set | §I.1 |
| `IntExpr` / $\mathcal{E}$ | symbolic integer algebra | §I.2, `preamble:20-91` |
| `CoordVar` $\chi_{s,k}$ | coordinate of axis $k$ of shape $s$ | §I.3, `preamble:45` |
| bounds lattice | monotone interval abstract interpretation | §I.2b, `preamble:1550-1551` |
| `IntAffineExprList` | affine stride chain $\sum \chi_k\sigma_k$ | §I.4, `preamble:138-140` |
| layout ladder | contiguous ⟷ strided ⟷ offset spellings | §I.4, `preamble:203-242` |
| `ElementStride` | liberal layout with `Any` singleton strides | §I.4, `element_stride.rs` |
| `injectivity-of` | monotone no-overlap proof token | §I.4, `preamble:4772-4797` |
| `IndexMap` / `int-subst-of` | view map + substitution composition | §I.5, `preamble:189-191, 3996` |
| substitution walk | compound-image copy primitive | §I.5, `subst_primitive.rs` |
| HLIR / `LogicalTensor` | logical dataflow spec | §I.6, `preamble:173-176` |
| LLIR / `LayoutTensorOp` | concrete implementation op | §I.6, `preamble:273-282` |
| match rule (`backend`) | HLIR→LLIR lowering via union | §I.6, `reference/.../match_functional.egg` |
| genome $G$ | per-class producer choice | §I.7, `extraction.rs:638` |
| `SamplingSpace` | SCC-aware acyclic sampling structure | §I.7, `extraction.rs:763` |
| `(1+λ)` ES | elitist mutation-only search | §I.8, `search.rs:356` |
| DPS / Poison | destination-passing rewrite | §I.9, `dps.rs` |
| bufferize | one-shot MLIR-style memory planning | §I.9, `bufferize.rs:1739` |

## Appendix B — The schedule, annotated

```lisp
(run-schedule
  (saturate (run prop))                                      ; seed shape/dtype/rank
  (saturate (saturate (run main_ruleset) (run prop))         ; ring + layout ladder + IndexMap,
            (run subst-walk))                                ;   subst as outer stratum; NO backend yet
  (saturate (run element-stride-discovery))                  ; (cuda-lite) discover ElementStride layouts
  (saturate (saturate (run main_ruleset) (run backend)       ; lower HLIR -> LLIR (backend match rules)
                      (run prop)) (run subst-walk))
  (run materializing-copy-mint)                              ; mint materializing copies
  (run layout-tensor-op-metadata)                            ; attach op input/output list metadata
  (saturate (run cleanup))                                   ; (cuda-lite) delete HLIR
  (saturate (run fixpoint-invariants)))                      ; terminal invariant stratum
```
Source: `crates/luminal_cuda_lite/src/bindings.rs:386`; core default
`src/extraction.rs:4644`. See §II.2.
