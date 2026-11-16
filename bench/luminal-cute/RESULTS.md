# Luminal-CuTe A/B — Phase 0: baseline ceiling (measurement harness)

Goal: a simple A/B to test whether a CuTe-layout IR gives Luminal a **higher
performance ceiling** on Metal, with Luminal's search graph on top. Phase 0
builds the measuring stick and establishes the baseline.

## Experimental design (agreed)

Two-step A/B that **controls for per-kernel quality so the delta isolates fusion**:

1. **Step 1 — tile the baseline.** Replace Luminal-Metal's naive matmul kernel
   (`MulReduceSumGeneric`, one thread per output element) with a tiled
   `simdgroup_matrix` kernel, *in the baseline*. Now both arms have good per-op
   kernels; the residual gap to peak is attributable to **fusion**, not tiling.
2. **Step 2 — fuse with CuTe.** The CuTe arm uses `cute-layout`
   (`refs/cuda-rust/cuda-oxide/cute-rs-preview/cute-layout`) to derive the tile /
   thread-value layouts, and an egglog `backend` rewrite to fuse a GEMM with its
   epilogue (keep the M×N intermediate resident instead of round-tripping global
   memory). A/B delta = the value of fusion.

CuTe's job: make the tiled intermediate layouts first-class and *statically
composable/verifiable* (composition + `logical_divide`), which Luminal's flat
affine-offset representation cannot express. Fusion **decisions** stay egglog
rewrites + search (CuTe is representation, not a fusion engine).

## Environment

- Apple M2, Metal 3, macOS 14.0, arm64.
- Rust `1.99.1` (luminal MSRV is 1.91; default stable here is 1.89 — use
  `cargo +1.99.1`).
- luminal clone at `refs/luminal` (gitignored); harness example at
  `crates/luminal_metal/examples/ab_ceiling.rs` (copy here as `ab_ceiling.rs`).
- Run: `cargo +1.99.1 run --release --example ab_ceiling -p luminal_metal`

## Baseline results (debug host build; `best_nanos` is GPU kernel time)

| N (=M=K) | gemm ms | GFLOP/s | gemm+relu ms | GFLOP/s | relu Δ | winning matmul op |
|---:|---:|---:|---:|---:|---:|---|
| 256  | 0.45  | 75  | 0.50  | 67  | +11% | `MulReduceSumGeneric` |
| 512  | 2.38  | 113 | 2.53  | 106 | +6%  | `MulReduceSumGeneric` |
| 1024 | 19.5  | 110 | 18.6  | 116 | −5%* | `MulReduceSumGeneric` |

\* −5% is measurement noise (separate stochastic search runs; ±5% GPU jitter).

## Findings

1. **Baseline GEMM ≈ 110–115 GFLOP/s ≈ ~3% of M2 FP32 peak (~3.6 TFLOP/s).**
   The matmul is the naive `MulReduceSumGeneric` (one thread per output element,
   codegen at `crates/luminal_metal/src/kernels.rs:528-681`). **~25–30× tiling
   headroom** → this is Step 1's target (a `simdgroup_matrix` tiled kernel).
2. **The relu epilogue is a *separate* set of kernels** (relu lowers to
   `LessThanGeneric` + `SelectFunctionalGeneric` + `ConstantGeneric` +
   `CastGeneric`, none fused into the matmul). The M×N intermediate round-trips
   global memory → this is the fusion opportunity Step 2 removes.
3. **Fusion's value is sub-noise at naive speeds.** The epilogue costs ≲ a few %
   of the naive GEMM, lost in jitter. It only becomes a measurable fraction once
   the GEMM is ~25× faster (tiled). **Conclusion: tile first (Step 1), then fuse
   (Step 2).** You cannot honestly A/B fusion on a naive baseline.

## Open methodology notes for the real A/B

- Current harness re-runs the full (stochastic) search per measurement → ±5%
  noise. For the tiled-vs-fused A/B, pin the extracted plan and re-time it N
  times (min), to get a low-noise delta.
- No MPS reference yet: `metal-0.31` only binds MPS **ray tracing**, not
  matrix-multiply. Add an empirical ceiling via a standalone objc
  `MPSMatrixMultiplication` rig (or compare the tiled kernel to it) in Step 1.

---

# Step 1 — tiled `simdgroup_matrix` GEMM (done)

**Change** (patch: `step1-tiled-gemm.patch`, against `crates/luminal_metal/src/kernels.rs`):
`reduce_product` now emits a tiled `simdgroup_matrix` kernel (8×8 cooperative
tiles, one simdgroup/32 threads per output tile, 2D threadgroup grid via the
existing `KernelLaunch{grid,block,shared_bytes}`) for the clean 2D-GEMM case
(F32, rank-3 product reduced over innermost K to rank-2 output, static dims ÷8,
operands broadcast-of-row-major with unit K stride — all detected from
`read_affine` strides). Falls back to the naive scalar loop otherwise. **Zero
egglog changes** — this is purely a codegen upgrade behind the existing
`MulReduceSumGeneric` op. The naive→tiled transpose uses `simdgroup_load(..,
transpose=true)` on B[N,K] to form the (k,n) operand.

**Correctness** (`tests/matmul_tiled.rs`): verified vs a CPU f64 reference on
non-square tile-aligned shapes (64×48×32, 24×72×128, max err ~3e-7) and the
non-÷8 fallback (12×20×12). Non-square shapes catch lda/ldb/transpose bugs.

**Performance** (best profiled plan; GPU kernel time):

| N (=M=K) | naive GFLOP/s | **tiled GFLOP/s** | speedup | tiled gemm ms |
|---:|---:|---:|---:|---:|
| 256  | 80  | **153** | 1.9× | 0.22 |
| 512  | 113 | **554** | 4.8× | 0.48 |
| 1024 | 110 | **769** | **7.0×** | 2.79 |

Tiled 1024³ = ~21% of the ~3.6 TFLOP/s peak (simplest possible tile; shared-mem
tiling would push higher, but not needed — both A/B arms share this GEMM).

**The fusion signal emerged.** With the GEMM ~7× faster, the relu epilogue's cost
went from sub-noise (−5% at naive speeds) to **28–55%** of the tiled GEMM
(1024³: gemm 2.79ms vs gemm+relu 3.62ms → +0.82ms ≈ 30%). The M×N intermediate
round-trip + the (decomposed) relu kernels are now a large, measurable fraction.
**This is exactly the headroom Step 2 (CuTe epilogue fusion) targets, now that
tiling has exposed it.** Confirms the agreed order: tile first, fuse next.

Caveat: numbers vary run-to-run (stochastic search + GPU jitter). For the Step 2
A/B, pin the extracted plan and re-time N× (min) for a low-noise fusion delta.
Part of the current "relu cost" is relu's own decomposition (`LessThan`+`Select`
+…), not pure round-trip; the Step-2 fused kernel will measure the attributable
saving directly.

---

# Step 2 — CuTe-driven epilogue fusion (GEMM + bias)  (done)

**Change** (patch: `step2-fusion.patch`): a new fused op `MatmulBiasGeneric`
(`crates/luminal_metal/src/ops/matmul_bias/`) that computes
`D[m,n] = sum_k A[m,k]*B[n,k] + bias[m,n]` in ONE kernel — the bias is applied in
the simdgroup epilogue (store the 8×8 C tile to threadgroup memory, add bias,
write out) so the M×N matmul intermediate never round-trips global memory.
- **egglog `backend` match rule** (`matmul_bias/match_functional.egg`): matches
  `LogicalAdd(reduce_sum(mul(a,b), axis), bias)`, F32, injective RM-contiguous
  output, and mints the fused op. The unfused `{matmul; add}` plan always remains,
  so fusion is a pure optimization the search elects when faster.
- **Fused kernel** (`kernels.rs::matmul_bias_codegen`): tiled simdgroup path for
  ÷8 F32, per-element fused fallback otherwise.
- **CuTe integration**: `cute-layout` (vendored from
  `refs/cuda-rust/cuda-oxide/cute-rs-preview/cute-layout` into
  `vendor/cute-layout`, a standalone workspace member — the cuda-oxide workspace
  is broken/missing members, and the crate has zero deps) is used in
  `kernels.rs::cute_verify_gemm_tiling`: A/B/D are modeled as CuTe `(shape:stride)`
  Layouts and the 8×8 tile geometry is **verified admissible** (unit stride where
  simdgroup load/store reads contiguously; even ÷8 division; CuTe's per-tile base
  offset `d.call((m0,n0))` must equal the kernel's `m0*ldc+n0`). This is the
  Luminal→CuTe seam in miniature: the kernel constants are cross-checked against
  the layout algebra, not trusted. (For this simple row-major case the CuTe
  offsets match the hand-rolled ones; the value is generality/verification as
  fusion gets more complex — swizzles, transposed epilogues, multi-level tiles.)

**Correctness + selection** (`tests/matmul_bias.rs`): the fused op is forced
(exclude `AddFunctionalGeneric`) and verified vs CPU reference — tiled 64×48×32
(err 4.8e-7) and per-element fallback 12×20×12 (err 2.4e-7). Both select
`MatmulBiasGeneric`.

**A/B** (`tests/fusion_ab.rs`, pinned-plan min-of-400-execute timing — both arms
share the identical tiled simdgroup GEMM; only the epilogue differs):

| M×N×K | regime | unfused ms | fused ms | speedup | saved |
|---:|---|---:|---:|---:|---:|
| 512×512×512   | compute-bound | 0.52 | 0.52 | 1.00× | −0.1% |
| 1024×1024×512 | compute-bound | 1.60 | 1.51 | 1.06× | 5.8% |
| 1024×1024×128 | mixed         | 0.72 | 0.55 | 1.32× | 24% |
| 1024×1024×64  | memory-bound  | 0.52 | 0.36 | **1.45×** | 31% |
| 2048×2048×64  | memory-bound  | 1.56 | 1.08 | **1.44×** | 31% |

**Finding:** epilogue fusion saves **~0 when compute-bound (big K), up to ~1.45×
(31%) when memory-bound (small K)** — exactly the physics: eliminating the M×N
intermediate round-trip matters only when it is a meaningful fraction of total
traffic. The confounded first run (85–93%) was *tiling*, not fusion — the search
had silently picked the naive decomposed matmul for the "unfused" arm; forcing
the tiled GEMM in both arms (exclude `ReduceSumGeneric`) gave the honest numbers.

**Thesis check:** CuTe-as-IR-enabling-fusion gives a higher ceiling — up to 1.45×
here for a *single* cheap bias epilogue. Richer fusion (heavier/compound
epilogues like the relu `LessThan`+`Select` chain, or attention's large resident
intermediates) is where CuTe's tiling algebra earns its place and the win grows.
The representation swap alone is neutral; the ceiling moves with the fused tiled
kernels CuTe makes expressible + verifiable.

## Measurement methodology notes

* Pinned-plan timing (compile once via registry filter, min over many
  back-to-back `execute()`) is REQUIRED to resolve single-digit % deltas; the
  search-based harness (`examples/ab_fusion.rs`) has ±several-% run-to-run noise.
* Arena budget (4 GB) limits shapes: a candidate can materialize the `[M,N,K]`
  broadcast product, so keep M·N·K·4 bytes under budget (e.g. 2048²×64 ≈ 1 GB OK;
  2048²×256 ≈ 4.3 GB fails). The fused/tiled streaming plan does not materialize
  it, but the stochastic search must *find* that plan — bump generation_size.

---

# THE BATTLEGROUND — CuTe layout vs affine layout, identical kernel (the real test)

Fusion-is-good was beside the point. The actual question: **given the exact same
kernel primitives, does a CuTe-layout Luminal beat a standard-affine-layout
Luminal?** Isolating this required finding where the two layout representations
genuinely differ.

## Why a global-layout swap is provably null

Luminal's offset algebra (`IntExpr`: Lit/Var/CoordVar/Add/Mul/TruncDiv/TruncRem/
CeilDiv/Min/Max) has **no bitwise/XOR operator**. For *global* integer-stride
layouts it is equi-expressive with CuTe — it can already express row/col-major,
strided views, broadcasts, and even **block-tiled** layouts (via TruncDiv/
TruncRem). So swapping the global layout representation changes nothing.

The **one** thing CuTe can express that Luminal's `IntExpr` cannot is an **XOR
swizzle** (CuTe's 𝔽₂ swizzle strides). Swizzles matter in exactly one place:
**threadgroup-memory bank conflicts.** (CuTe's other non-affine form, ℤˢ
coordinate layouts, is for NVIDIA TMA — irrelevant on Metal.)

## Atomic kernel demo (`examples/swizzle_demo.rs`, Apple M2)

A kernel stages a tile in threadgroup memory and reads it in a column pattern
that 32-way bank-conflicts under a plain layout (BK=32 banks). The SAME kernel
template is instantiated with three shared-memory index functions — the only
difference:

| layout | index | time (ms) | vs plain | tgmem | expressible by |
|---|---|---:|---:|---:|---|
| plain   | `row*BK + col` | 60.4 | 1.00× | 4096 B | Luminal `IntExpr` |
| padded  | `row*(BK+1) + col` | 23.4 | 2.58× | 4224 B | Luminal `IntExpr` |
| **swizzle** | `row*BK + (col ^ (row&31))` | **21.2** | **2.85×** | **4096 B** | **CuTe only** |

Stable across runs (min of 32, ±0.1 ms). The swizzle index is **generated by
`cute_layout::Swizzle::new(5,0,5)`** (verified in-demo to equal the kernel's XOR),
proving it comes from CuTe's algebra, not a hand-rolled constant.

## Result

- Bank conflicts are real and dominant on M2: plain is **2.85× slower**.
- **CuTe's best (swizzle, 21.2 ms) beats affine's best (padded, 23.4 ms) by ~11%**
  — AND uses less threadgroup memory (4096 vs 4224 B).
- The win is attributable **100% to the layout representation**: identical kernel
  template; the swizzle is the single layout Luminal's `IntExpr` cannot express.

So: **a CuTe-layout Luminal has a strictly higher ceiling than a standard-layout
Luminal, given identical kernels** — because it can select a shared-memory layout
(XOR swizzle) the affine representation cannot, and that layout is both faster and
more memory-efficient (the memory saving compounds at the tgmem capacity limit:
swizzle fits a larger tile than padding).

## What this does and does not establish

- DOES: the representational gap is real, CuTe-exclusive, and worth a measurable
  win at the kernel level, generated from CuTe's own `Swizzle`.
- DOES NOT YET: wire the swizzle into Luminal's e-graph as a search-selectable
  threadgroup-memory layout. That is the next step — extend the layout rep to
  model intra-kernel (threadgroup) layouts so the search chooses {plain,padded}
  (affine arm) vs {plain,padded,swizzle} (CuTe arm), and the full system picks
  the swizzle. The kernel-level ceiling gap (this demo) is the precondition that
  makes that integration worth doing.

---

# CuTe-layout → MSL lowering (built)  +  full NVIDIA reference in refs

Decision: keep Metal, build the CuTe→MSL lowering ourselves (there is no CuTe→
Metal codegen upstream — NVIDIA's CuTe lowering targets PTX/NVVM). The second
paper (Cypress, arXiv:2504.07004) does NOT lower a CuTe IR — verified: its IR is
an event-based IR lowered to CUDA C++, and CuTe appears only as a leaf Tensor-
Core dispatch. The real CuTe→code lowering is NVIDIA's `cuda-rust`.

## NVIDIA reference pulled into refs
`refs/cuda-rust` was a sparse `blob:none` partial clone (only `cute-layout` on
disk). Ran `git sparse-checkout disable` to materialize the full tree. Now
present: the complete CuTe→CUDA lowering stack —
`cuda-oxide/cute-rs-preview/{cute-rs, dialect-cute, cutlass-primitives}`, the
`cuda-oxide/crates/` MLIR pipeline (`dialect-ptx`, `dialect-nvvm`, `dialect-mir`,
`mir-importer`, `cuda-oxide-mlir-export`, `rustc-codegen-cuda`, `llvm-export`),
and `cutile-rs/{cutile-ir, cutile-compiler}` (the Tile IR). This is the reference
for how NVIDIA structures a CuTe/tile IR and lowers it — to adapt for MSL.

## cute_msl: the lowering (`crates/luminal_metal/src/cute_msl.rs`)
Given a `cute_layout` `Layout` (shape:stride) and optional `Swizzle`, emit the
MSL offset expression:
- `layout_offset(layout, coords)` -> `crd2idx` as `sum_i coord_i * stride_i`
  (stride-0 axes drop, stride-1 axes emit the bare coord).
- `swizzle_offset(sw, base)` -> the XOR `base ^ ((base & y_mask) >> S)`, matching
  `Swizzle::apply` (unit-tested to agree numerically for all coords).

So kernel addressing is GENERATED from CuTe layout objects, not hand-written.
The swizzle benchmark is now CuTe-IR-driven — the three index expressions come
out of `cute_msl`:
```
   plain: As[(row * 32 + col)]                                      # affine (also Luminal IntExpr)
  padded: As[(row * 33 + col)]                                      # affine
 swizzle: As[((row * 32 + col) ^ (((row * 32 + col) & 992u) >> 5u))] # CuTe-only (from Swizzle::new(5,0,5))
```
Timings reproduce the hand-written version (plain 60.4 / padded 23.6 / swizzle
21.3 ms), confirming the generated MSL is faithful. This is the atomic unit of
"CuTe as the kernel IR" for Metal: the layout object drives the codegen.

NEXT: route the GEMM / matmul_bias kernels' addressing through `cute_msl`;
extend to nested (hierarchical) layouts via CuTe `crd2idx` over nested coords;
then wire a swizzled shared-memory layout into the e-graph as a search-selectable
choice (the full-system CuTe-vs-affine A/B).

---

# GEMM kernels routed through cute_msl + nested-layout support (done)

## 1. GEMM addressing is now CuTe-generated
`matmul_simdgroup` (the tiled GEMM) and `matmul_bias_codegen` (fused GEMM+bias)
no longer hand-write index arithmetic. They build CuTe `Layout`s —
A[M,K]:[lda,1], B[N,K]:[ldb,1], C[M,N]:[ldc,1], bias[M,N]:[bias_sm,bias_sn] — and
emit all offsets through `cute_msl::layout_offset` (the simdgroup_load/store base
offsets and the epilogue bias-read / output-write). Tile/loop coords are `ulong`
for 64-bit-safe crd2idx. A broadcast [N]/[M] bias drops its zero-stride axis
automatically. Correctness unchanged: `matmul_tiled` (3) and `matmul_bias` (2)
tests pass.

## 2. Nested-layout support in cute_msl
`cute_msl` now lowers HIERARCHICAL CuTe layouts via `crd2idx` over nested
coordinates (CuTe's offset is multilinear: `sum over leaves of leaf_coord *
leaf_stride`):
- `Coord { Leaf(name) | Nest(children) }` — a natural coordinate tree.
- `layout_offset_tree(layout, coord)` — recurses the stride tree and coord tree
  in lockstep (congruence-checked) and emits the summed offset.
Verified (`cute_msl` unit tests):
- nested `((2,3),4):((1,8),2)` → `(a + b*8 + c*2)`, matches CuTe `.call` for all
  natural coords;
- a **tiled** `((4,8),(4,8)):((256,32),(8,1))` (32×32 matrix as 8×8 tiles)
  → `(ti*256 + mi*32 + tj*8 + nj)`, equal to the flat row-major address
  `(ti*8+mi)*32 + (tj*8+nj)` for every tile coordinate — the real tiling use;
- congruence mismatches error.

This is the piece CuTe's layout algebra exists for: a tiled view expressed as one
nested layout, lowered to the correct address automatically. Next: express the
GEMM operands as nested tiled layouts and drive the loop structure from them;
then wire a swizzled shared-memory layout into the e-graph as a search choice.

---

# #1 nested-layout GEMM (done) + an HONEST swizzle finding

## Register-tiled SGEMM, addressing fully CuTe-generated (`examples/tiled_gemm.rs`)
C=A@B^T, BM=BN=64, BK=8, TM=TN=4, 256 threads, shared-memory K-strip staging.
ALL addressing is generated by `cute_msl`: the shared-memory index, the global
A/B loads, and — the point of #1 — the per-thread register-tile OUTPUT write via
NESTED `crd2idx` over the tiled C layout `((TM,BM/TM),(TN,BN/TN))`. Correct vs CPU
reference (err 3e-7). Nested layouts are now load-bearing in a real GEMM loop.

## The honest finding: the swizzle does NOT help this GEMM
| smem layout | GFLOP/s | tgmem | expressible by |
|---|---:|---:|---|
| plain   | 353 | 4096 B | Luminal IntExpr |
| padded  | 377 | 4608 B | Luminal IntExpr |
| swizzle | 244 | 4096 B | CuTe only |

- **Padding beats plain by only ~7%** → the ~7% is the ENTIRE conflict-avoidance
  ceiling; bank conflicts are not the bottleneck at this ~350 GFLOP/s (~10% of
  peak) optimization level.
- **The swizzle LOSES** (244): its per-access XOR overhead, recomputed in the hot
  inner loop, costs more than the small conflict it removes.
- Contrast the swizzle_demo (2.85× plain→swizzle): that was a PATHOLOGICAL pattern
  (512 re-reads of a 32-wide column) engineered to make conflicts dominate. A real
  GEMM's access pattern is far more benign.

## What this means for "swizzle-into-search → Luminal-CuTe tied with CuTe"
- Representational parity is REAL: CuTe expresses the XOR swizzle, Luminal's
  IntExpr cannot. That part of the thesis holds.
- But the PERFORMANCE lever does not bite here: at achievable Metal-GEMM
  optimization levels, conflicts aren't the bottleneck, so wiring the swizzle
  into search would have the search correctly pick plain/padded — i.e.
  Luminal-CuTe would TIE affine-Luminal on these GEMMs, not beat it. The swizzle
  wins only in genuinely conflict-bound kernels (the demo, or a near-peak GEMM
  where the shared reads are vectorized / computed once and conflicts dominate).
- The one defensible structural edge of swizzle over padding is MEMORY: swizzle
  avoids the pad column, so at the shared-memory capacity limit it fits a larger
  tile than padding → higher occupancy. Demonstrating that needs a near-peak GEMM.

## Options
1. Accept the honest neutral result for Metal GEMMs at this level (swizzle is a
   niche lever; representational parity without a GEMM speed win).
2. Build a near-peak Metal SGEMM (vectorized loads, double-buffering, bigger
   tiles, overhead-free swizzle addressing) where conflicts bottleneck and the
   swizzle's memory efficiency enables bigger tiles — a major kernel effort.
3. Find a different workload/capability where CuTe gives a real Metal win.

---

# Mechanically-constructed kernel: cute-layout algebra BUILDS the GEMM layouts (done)

The clarifying distinction (resolved with the user): CuTe and Luminal's affine
`IntExpr` are equi-EXPRESSIVE (both can denote + lower any affine/block-tiled
address map — Luminal already lowers its own layouts via `kernels.rs::
layout_read_index`). They are NOT equi-CONSTRUCTIVE: CuTe has a callable,
validated layout ALGEBRA (`logical_divide`/`zipped_divide`/`blocked_product`/
`make_layout_tv`/`composition`/`complement`/`inverse`, all `Result<Layout>`) that
mechanically BUILDS tiled + thread-partitioned layouts. Luminal has no such
algebra (its layout construction lives inside egglog saturation, geared to
search/discovery; its layouts model only the global element offset, not the
intra-kernel tile/thread hierarchy). That construction algebra — not
expressiveness — is the reason to adopt CuTe. Swizzles were never the point.

## Verified construction (`examples/cute_construct.rs`)
For the GEMM output write (BM=BN=64, TM=TN=4, 256 threads):
- `make_layout_tv(thr=(16,16):(16,1), val=(4,4):(4,1))` -> `tv` mapping
  `(thread,value) -> tile cell`.
- `composition(tile_c=(64,64):(512,1), tv)` -> `out_tv` mapping
  `(thread,value) -> within-tile global offset`.
- `zipped_divide(C, [BM,BN])` -> the tile-grid strides.
- **Numerically verified**: `out_tv.call` equals the hand formula
  `(ty*TM+i)*N + (tx*TN+j)` for ALL 4096 (thread,value) pairs.
- `cute_msl` lowers the CONSTRUCTED layouts to MSL:
  within-tile `(tx*4 + ty*2048 + j + i*512)`, tile-base `(tj*64 + ti*32768)` —
  summing to the exact hand address, with NO hand-derived strides. (cute's
  natural coord is inner-first per mode: `((tx,ty),(j,i))`.)

## Wired into the running GEMM (`examples/tiled_gemm.rs`)
The register-tiled SGEMM's output addressing is now produced by
`make_layout_tv` + `composition` + `cute_msl`, not hand-written. The GEMM runs
correct (err 3e-7), ~350 GFLOP/s (unchanged — same address, mechanically built).

This is the first genuinely mechanically-constructed Metal kernel: cute-layout's
algebra builds the tile/thread layout; cute_msl generates the MSL addressing.
NEXT: construct the A/B global + shared-memory tile layouts the same way (so the
WHOLE kernel comes from a few layout ops), then make tile config a parameter the
search varies (cute-layout builds each config's kernel) -> "CuTe IR + Luminal
search" autotuning tiled kernels.

---

# Full A/B/C layout set constructed by the algebra (done)

Extending the mechanically-constructed GEMM so ALL per-thread fragment addressing
comes from cute-layout algebra (verified in `examples/cute_construct.rs`):
- **C output** (TV partition): `make_layout_tv(thr,val)` + `composition(tile_c,tv)`
  → `(thread,value) -> within-tile global offset`. Tile base: `zipped_divide(C,
  [BM,BN])`.
- **A read fragment**: `zipped_divide((BM,BK):(stride_a,1), [TM,BK])` →
  `((TM,BK),(BM/TM,1))`, coord `((i,k),(ty,0))` = `As[ty*TM+i][k]`. Verified for
  all (ty,i,k); adapts automatically to the shared row stride (plain BK / padded
  BK+1).
- **B read fragment**: `zipped_divide((BN,BK):(stride_b,1), [TN,BK])`, coord
  `((j,k),(tx,0))` = `Bs[tx*TN+j][k]`.
All lowered to MSL by `cute_msl`. The GEMM runs correct (err 3e-7): plain 280,
padded 363 GFLOP/s.

## Honest lowering-quality finding
plain dropped 345 → 280 GFLOP/s because `cute_msl`'s naive `sum coord*stride`
emits `i*8 + k + ty*32` (two multiplies) where the hand form `(ty*4+i)*8 + k`
folds the common factor (one multiply). This is a CODEGEN-QUALITY issue, not a
construction one — the constructed layouts are correct; the lowering is naive.
Fixes, both standard CuTe: (a) `coalesce` the fragment layout so adjacent modes
with contiguous strides merge (ty,i -> one (ty*TM+i) coord at stride BK); (b)
SLICE the fragment by the thread coord (CuTe's partial evaluation) so the
thread offset is a base pointer computed ONCE and the inner loop indexes only
(i,k). Slicing is the idiomatic GEMM path and the clear next step.

---

# Slicing closes the lowering-quality gap (done)

Added `cute_msl::slice` — CuTe partial evaluation (Cecka §2.5.1). A `Coord` leaf
is now `Leaf` (free/loop) or `Fixed` (sliced/loop-invariant); `slice(layout,
coord)` returns `(base, free)` where `base` is the fixed coords' contribution
(computed ONCE — a per-thread base pointer) and `free` is the offset over the
loop coords. `base + free == crd2idx` (unit-tested).

Wired into `tiled_gemm`: each A/B register fragment is sliced by its thread coord
->  `As_thr = As + (ty*32)` hoisted before the k-loop, inner loop indexes only
`As_thr[i*8 + k]`. This recovers (and beats) hand-written inner-loop efficiency:

| smem | before slice | after slice | original hand-written |
|---|---:|---:|---:|
| plain  | 280 | **337** | ~345–353 |
| padded | 363 | **384** | ~350–377 |

Correct (err 3e-7). cute_msl (6), matmul_tiled (3), matmul_bias (2) tests pass.

So the GEMM is now a **no-regret mechanical construction**: every address is built
by cute-layout algebra (make_layout_tv / composition / zipped_divide), lowered by
cute_msl with slicing + nested crd2idx, and runs at hand-written speed. This is
what Luminal's layout representation cannot do (no constructive algebra). NEXT:
make the tile config (BM/BN/BK/TM/TN) a parameter — cute-layout rebuilds every
layout from it, cute_msl regenerates the kernel — and let Luminal's search
autotune the config: "CuTe IR + Luminal search".

---

# Tile config as a search parameter — the payoff (done)

`examples/gemm_autotune.rs`: the ENTIRE GEMM kernel is generated from a config
`(BM,BN,BK,TM,TN)` — cute-layout's algebra (make_layout_tv/composition/
zipped_divide) rebuilds every layout from the config, cute_msl lowers them
(nested crd2idx + slicing). So each config yields a correct kernel for free. The
search then does what Luminal's search does: profile every candidate on the GPU,
keep the fastest.

1024x1024x1024 on M2, 105 valid configs searched:
- baseline (BM64 BN64 BK8 TM4 TN4): 611.7 GFLOP/s (3.51 ms)
- autotuned best (BM128 BN64 BK8 TM4 TN8): 728.5 GFLOP/s (2.95 ms)
- **1.19x from autotuning the tile config**, zero manual kernel edits.

The magnitude is modest, but the mechanism is the point: hand-writing 105 tiling
variants is infeasible; mechanical construction makes 105 correct kernels from
105 configs trivial, and a profiling search picks the best. That IS "CuTe IR +
Luminal's search graph on top" — the original thesis — realized at the kernel-
generation level. It is Luminal's profiling METHODOLOGY (profile, keep best); not
yet wired into Luminal's e-graph/genome search as a config dimension. That
integration (config-variant ops the genome selects, or a backend that autotunes
config during matmul codegen) is the next step.

---

# Tile config wired into Luminal's e-graph / genome search (done)

The autotuner above was Luminal's *methodology* bolted on the side. This step
makes the config a real dimension of **Luminal's own e-graph search**: the tile
config is now a child of a backend op, and the genome selects it.

## How the config becomes a search choice
A new backend op `MatmulTiledGeneric` (`crates/luminal_metal/src/ops/matmul_tiled/`):
- **egglog match rule** (`match_functional.egg`) matches the fused dot
  `reduce_sum(mul(a,b), axis)` with an F32 injective RM-contiguous output (same
  pattern `matmul()` lowers to), then mints **one producer enode per tile config**:
  ```
  (union ?op (LayoutTensorOpMatmulTiledGeneric ?a_lt ?b_lt ?out_layout 0))
  (union ?op (LayoutTensorOpMatmulTiledGeneric ?a_lt ?b_lt ?out_layout 1))
  (union ?op (LayoutTensorOpMatmulTiledGeneric ?a_lt ?b_lt ?out_layout 2))
  (union ?op (LayoutTensorOpMatmulTiledGeneric ?a_lt ?b_lt ?out_layout 3))
  ```
  The last constructor child is the config index (into `MATMUL_CONFIGS`). All
  four are unioned into the op class, so the matmul's output e-class has **four
  distinct producer enodes differing only in the config**. This is exactly the
  LayoutTensorOp producer-choice the genome already samples — the config is now
  one of those choices, no new search machinery.
- **matcher** (`MatmulTiledMatcher`) extracts the config from the enode's i64
  child (`site.child_i64(3)`) → `MatmulTiled { config }`.
- **codegen** (`kernels.rs::matmul_tiled_codegen`) generates the register-tiled
  shared-memory GEMM for that config — **all addressing built by cute-layout's
  algebra** (`make_layout_tv`/`composition`/`zipped_divide`) + `cute_msl`
  (nested crd2idx + slicing), adapted to the Luminal DPS ABI
  (`k(a,b,out,params,tid,bid)`, lda/ldb from `read_affine`, ldc=N).

So the genome picks a config enode → that config's kernel is generated → the GA
profiles it on the real GPU → keeps the fastest config. The config search is now
**inside** Luminal's `(1+λ)` genome search, not a separate harness.

## Verified (`tests/config_search.rs`)
256³ F32 matmul, competing producers (`MulReduceSumGeneric`, `ReduceSumGeneric`)
excluded so the output class's only producers are the four config variants → the
genome **must** select a config. Result: the search selects a `MatmulTiledGeneric`
config and the GEMM is correct vs CPU reference (**max err 5.6e-8**).

## The contract fix this exposed (the honest part)
Adding config-variant producers to the live registry broke two existing tests —
and the root cause was a real bug: Luminal's extraction contract is **"a minted
enode must be codegen-able"** (presence = valid), but both new ops (`matmul_tiled`
now, `matmul_bias` from Step 2) **bailed at codegen** for geometries their match
rules allow. A bailing producer perturbs/starves the stochastic genome search
(with few generations it can fail to find any valid plan — e.g. deep_fork_join's
`projected + projected` matches matmul+bias but is a *non-transposed* A@B the
clean path rejected → "no candidate genome produced an executable plan").

Fix — make both ops **never bail**, like the existing `mul_reduce_sum` (which
always falls back to the naive loop):
- `matmul_tiled_codegen`: invalid config / geometry → falls back to
  `reduce_product` (the general matmul). Every config enode is codegen-able.
- `matmul_bias_codegen`: non-clean geometry → new `matmul_bias_naive_codegen`, a
  general per-element fused kernel built from each operand's **full affine
  coefficients** (`read_affine` → `cute_msl::layout_offset`), so A@B, A@B^T, and a
  computed/broadcast bias all lower correctly — not just the restricted
  broadcast-row-major A@B^T.

Full `luminal_metal` suite green afterward (22 test binaries, 0 failures).

## What this establishes
The tile config — the autotuning lever from the previous section — is now a
first-class **search dimension inside Luminal's e-graph**: cute-layout mechanically
builds each config's kernel, the egglog match rule exposes the configs as producer
enodes, and Luminal's genome search profiles and selects among them. "CuTe IR +
Luminal's search graph on top" is realized end-to-end: the construction algebra
makes the kernel variants, Luminal's search picks the winner.
