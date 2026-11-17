# Hard rules (written in stone)

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

## 2. Never modify Luminal core

Luminal's source stays pristine — it is the affine baseline. The cute path is
external and additive, on Luminal's public API only. When blocked, surface it to
the user; never invent a workaround.
