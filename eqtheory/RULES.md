# Hard rules (written in stone)

All of these are enforced by `build.rs` and **fail the build** (not just a test) on
any violation — `cargo build` will not succeed while a rule is broken.

## 4. No hand-constructed graphs in comparisons — models are LOADED

A model under comparison must come from a loader (a real architecture / checkpoint,
e.g. Luminal's `model_zoo`), never authored inline with `Graph::new` and the frontend
builders (`.matmul`, `.split_dims`, `.merge_dims`, `.softmax`). If the test authors the
graph, it can shape the model to whatever makes a point — that is the gaming this bans.

Enforcement (`build.rs`): `Graph::new` and those builders may appear only in `loaders.rs`.
Everywhere else they fail the build.


## 1. Kernel sets must match at all stages

Any A/B that compares the **cute** layout theory against the **affine** layout
theory MUST hold the kernel set identical across both arms, at every stage of
execution. Both arms select from ONE shared kernel library. Neither arm may use a
bespoke kernel the other cannot. The matmul kernel, the softmax kernel, the
copy/materialize kernel — byte-identical for both.

The ONLY thing allowed to differ between the two arms is what the layout theory
decides: which operands are read by stride vs materialized by a copy, and the
stride arguments passed to the shared kernels. If the kernel implementations
differ, the comparison measures kernels, not layout theories, and is void.

Consequence: Luminal's own backend is NOT a fair perf baseline — its kernels are
its own and cannot be shared (Luminal is never modified, see rule 2). Luminal is a
**correctness oracle** only. The fair perf comparison is cute-layout vs
affine-layout, both dispatching the shared kernel library in `src/metal_backend.rs`.

Enforcement: `tests/kernel_set.rs` — Metal kernel/library/pipeline creation may
appear in `src/metal_backend.rs` ONLY. No other source file may define or compile a
kernel. One kernel source → both arms provably share the set.

## 3. No arm-run hacks — both arms come from the actual pipeline

Both arms of an A/B (cute-layout, affine-layout) must be produced by the ACTUAL
pipeline: lower → compose → canonicalize → match a kernel per equivalence class →
dispatch. The arm is selected by a `Ruleset` enum applied *through* canonicalization,
never by hand-branching the kernel dispatch on a per-arm `bool` inside a test.

A test that writes `if affine { materialize(...) } else { strided(...) }` is a hack:
it hard-codes what the arm "would" do instead of letting the pipeline decide, and it
can silently build a strawman for the other arm. Forbidden.

Enforcement: `tests/no_arm_hacks.rs` — no source file may contain an `affine: bool`
(or `arm: bool` / `cute: bool`) dispatch flag.

## 2. Never modify Luminal core

Luminal's source stays pristine — it is the affine baseline. The cute path is
external and additive, on Luminal's public API only. When blocked, surface it to
the user; never invent a workaround.
