# CuTe as Luminal's Layout Equational Theory

## Thesis

Luminal is an equality-saturation compiler. It lowers tensor operations into an
e-graph, where each e-class is the set of equivalent ways to compute a value.
What lives in a class is determined entirely by the **equivalence relations**
(rewrite rules) the e-graph is given.

CuTe's layout algebra is an **equational theory for layouts** — a closed,
canonicalizing set of layout identities. It is a strictly more powerful and more
compact equational theory than the one Luminal uses today. Installed as the
e-graph's layout rewrite rules, it **compresses** layout representations that
Luminal's current theory blows up on, and makes more succinct and more fused
layouts reachable by extraction.

The value of CuTe is this equational theory. It is not a kernel generator, not a
tile autotuner, and not a faster per-kernel address computation. Any claim framed
in those terms is out of scope for this project.

## The problem: Luminal composes layouts by substitution, and it blows up

Luminal's layout theory (`src/layout/element_stride_rules.egg`) represents a
layout as a flat per-axis stride list and composes views (`LogicalIndexMapApply`)
by **coordinate substitution** (`int-subst-of`): it decomposes a flat index into
coordinates with `TruncDiv`/`TruncRem`, re-expresses them for the next view, and
substitutes. This grammar handles permutations, broadcasts, zero selections and
diagonals, and "deliberately stalls" on anything else.

Substitution is not closed. Composing a chain of reshapes/transposes nests the
index expression at every step, and because a reshape references its argument in
two places (`TruncDiv` and `TruncRem`), substitution **duplicates** the argument
each round. The term grows exponentially in chain depth. Luminal's own codebase
confirms this: `index_expr.rs` carries "exponential-blowup protection",
`egglog_utils/base.rs` notes "a combinatorial blow-up on the deep orderings",
and `repro/subst_layout_blowup.egg` is a ViT (conv → flatten → transpose →
class-token → position-add) regression whose purpose is to prove the substitution
reaches a fixpoint at all.

## The solution: CuTe composition is closed and compresses

CuTe's layout algebra (`cute-layout`, vendored here) provides layout identities
Luminal has no rule for:

- `composition(A, B) = C`: layout composition is closed; `A∘B` is another compact
  `Shape:Stride` layout, not a growing index expression.
- `coalesce(L) = L'`: merges contiguous-stride modes into a canonical minimal form.
- the `logical_divide` / `zipped_divide` / `blocked_product` tiling laws.
- `complement`, `left_inverse`, `right_inverse`.

Under this theory the composition of an arbitrarily deep view chain stays a
bounded, canonical layout. The blowup Luminal guards against is definitionally
absent, because the theory is closed under composition and `coalesce` canonicalizes.

## Evidence (`src/compression.rs`)

A ViT-style flatten/transpose view chain over N=4096, measured in the e-graph's
own currency (term size):

| rounds | substitution term nodes | cute layout nodes | same map |
|---|---|---|---|
| 1 | 7 | 6 | yes |
| 4 | 91 | 6 | yes |
| 8 | 1531 | 6 | yes |
| 12 | 24571 | 6 | yes |

Substitution grows exactly exponentially (every round roughly doubles the term).
The CuTe composed-and-coalesced layout stays constant at 6 nodes. Both denote the
identical address map, verified over all 4096 indices. The compression is
unbounded: constant versus exponential.

## The work: install CuTe's theory as the e-graph's layout rewrite rules

1. Represent CuTe layouts (`Shape:Stride`, nested) as e-graph terms.
2. Add rewrite rules for `composition`, `coalesce`, and the tiling/inverse laws
   that `union` equivalent layout terms, replacing the substitution ladder as the
   mechanism for composing `LogicalIndexMapApply` chains.
3. Bridge: a CuTe layout term and the corresponding Luminal layout term denote the
   same address map, so the new rules interoperate with the rest of the e-graph.

The ruleset is a **compile option** so the two theories can be deltaed cleanly:
run identical input under `Affine` (no CuTe rules) versus `Cute` (CuTe rules) and
compare the resulting e-graph.

## Harness (`src/egraph.rs`)

CuTe layouts are egglog terms (`CLayout`, `Modes`), run on the same egglog fork
Luminal uses, so the rules port into the core. A `Ruleset` enum selects the
theory. The first installed rule is CuTe's `coalesce` (merge adjacent modes when
`s1 == e0*s0`), verified address-preserving against `cute-layout` before use. On a
contiguous k-mode layout:

| modes | affine term modes | cute term modes |
|---|---|---|
| 2 | 2 | 1 |
| 8 | 8 | 1 |
| 32 | 32 | 1 |

Same input, ruleset toggled: `Cute` collapses the layout to its canonical single
mode; `Affine` leaves it uncompressed.

The `composition` rule closes the theory under `∘`. It is a **Rust primitive**
(`cute-compose`), backing egglog with `cute-layout::composition` + `coalesce` the
way Luminal backs `int-subst-of` with a Rust primitive. A `Compose` term rewrites
to the primitive's result under the `cute` ruleset; the primitive is verified
address-preserving against `cute-layout` before use. On the ViT-style view chain
(the one `subst_layout_blowup.egg` grows exponentially under substitution):

| rounds | affine Compose nodes | cute Compose nodes |
|---|---|---|
| 2 | 1 | 0 |
| 8 | 7 | 0 |
| 12 | 11 | 0 |

Under `Cute` the entire chain closes to a single compact layout literal (zero
`Compose` nodes); under `Affine` it stays an unreduced composition. This is the
closed, compressing composition living in the e-graph, toggled cleanly by the
ruleset compile option.

## Ported into the core behind a compile switch (`core-port/`)

The theory installs into Luminal's core mechanically, mirroring `element_stride`
exactly. A `cute-eqtheory` cargo feature is the compile-time switch; with it on:

- `cute_eqtheory::register(egraph)` registers the `cute-compose` primitive — one
  line added to `new_egraph()`, the same place `element_stride::register` is called.
- `cute_eqtheory::snippets()` splices the CuTe datatype + compose rule — one line
  added to `assembled_program_for()`, the same place `element_stride::snippets()`
  is collected.
- `cute_eqtheory` is declared with the same `#[path]` form as `element_stride`.

Total wiring: four lines plus one module, all `#[cfg(feature = "cute-eqtheory")]`,
so the default build is byte-unchanged. The in-core test
`cute_compose_closes_a_chain_in_core` builds the real `new_egraph()` +
`assembled_program_for()` and reduces a `CuteCompose` chain to the compact
composed layout — proving the theory is live in the core's own e-graph. The
inclusion was trivial, which is the architecture passing its own test.

## Head-to-head in the core, each theory in its own language

The two theories are never bridged — they are different e-graph languages for
layouts, and the compile switch selects which one the compiler speaks. The thesis
delta is measured by running the same workload under each theory, in its own
terms, and comparing e-graph size (the metric Luminal's own blowup regression
uses).

The in-core test `head_to_head_vit_chain_rows`:

- **Affine**: loads the real `subst_layout_blowup.egg` ViT chain and runs the
  affine theory (`prop` / `main_ruleset` / `subst-walk`). Composing those views by
  substitution grows the e-graph to **6,396 rows** at 24 generations (its full
  convergence is ~36,000, capped by the exponential-blowup protection at 40,000).
- **Cute**: the equivalent-depth view chain written as `CuteCompose` terms, run
  under the `cute` ruleset — **84 rows**.

**76x fewer rows** (≈430x at full affine convergence). Same workload, two
languages, no cross-talk — the closed composition theory does in 84 e-graph rows
what the substitution theory needs tens of thousands (and blowup protection) for.

## Endpoint

The compile switch's destination is the real fork: under `cute-eqtheory` the
frontend emits CuTe layout terms and the pipeline consumes them, replacing the
affine theory outright — not bridging to it. The head-to-head above is the
measurement that justifies that swap.

## What to measure

- **Compression**: term size of the composed layout under CuTe rules versus under
  the substitution ladder, on real view chains (the ViT repro and beyond). This is
  the primary metric. It is a property of the e-graph, not of any kernel.
- **Reachability**: fused or succinct layouts present in an e-class under CuTe
  rules that are absent under the affine rules.
- **Saturation cost**: whether the e-graph reaches a fixpoint without blowup
  protection once composition is closed.

## Non-goals

- Kernel tile autotuning, and any "best-of-N configs beats a fixed config" result.
- Standalone, hand-written kernels timed outside the e-graph search.
- Per-kernel execution-speed comparisons. CuTe's layout algebra and Luminal's
  affine addressing are equi-expressive and emit equivalent kernels; the thesis is
  about the equational theory, not kernel throughput.

## Methodology constraints

- Project code carries no comments. `tests/no_comments.rs` enforces this over
  `src/` and `tests/`; the vendored `cute-layout/` is excluded (it is not our code).
- Experiments report e-graph quantities (term size, reachability, saturation),
  not kernel timings.
- Every equivalence a rule asserts must be numerically verified against the
  address map it claims to preserve.
