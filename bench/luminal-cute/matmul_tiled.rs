//! Correctness of the tiled `simdgroup_matrix` GEMM fast path (Step 1 of the
//! Luminal-CuTe investigation). Verifies the tiled kernel against a CPU
//! reference on a NON-SQUARE, tile-aligned shape (square shapes hide lda/ldb/
//! transpose bugs), and that a non-tile-aligned shape still computes correctly
//! via the naive fallback.
#![cfg(target_os = "macos")]
mod support;

use luminal::{dtype::DType, graph::Graph};
use luminal::bufferize::BufferNode;
use luminal_metal::{MetalRuntime, harness_search_options};
use support::TestTransfers;

/// C[m,n] = sum_k A[m,k] * B[n,k], computed in f64 on the host.
fn reference(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            for ki in 0..k {
                acc += a[mi * k + ki] as f64 * b[ni * k + ki] as f64;
            }
            c[mi * n + ni] = acc as f32;
        }
    }
    c
}

fn run_matmul(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>, bool) {
    let mut g = Graph::new();
    let a = g.tensor(vec![m, k], DType::F32);
    let b = g.tensor(vec![n, k], DType::F32);
    let out = a.matmul(b.t());

    // Distinct, bounded, deterministic values (not symmetric, to catch swaps).
    let av: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let bv: Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let want = reference(m, n, k, &av, &bv);

    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &[(a.id, av.clone().into()), (b.id, bv.clone().into())]
            .into_iter()
            .collect(),
        &harness_search_options(),
    )
    .unwrap();

    // Did the matmul path run at all (fused MulReduceSum)?
    let used_fused = rt.plan().unwrap().dag.node_weights().any(|node| {
        matches!(node, BufferNode::Compute { op, .. } if op.label() == "MulReduceSumGeneric")
    });

    let mut arena = support::allocate_arena(&rt).expect("allocate execution arena");
    rt.upload(&mut arena, a.id, av).unwrap();
    rt.upload(&mut arena, b.id, bv).unwrap();
    rt.execute(arena.buffer()).unwrap();
    let got = rt.read_f32(&arena, out.id).unwrap();
    (got, want, used_fused)
}

fn assert_close(got: &[f32], want: &[f32], tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: length mismatch");
    let mut max_err = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        max_err = max_err.max(err);
        assert!(
            err <= 1e-3 + 1e-3 * w.abs(),
            "{tag}: element {i}: got {g}, want {w} (err {err})"
        );
    }
    eprintln!("{tag}: OK (max abs err {max_err:.2e})");
}

#[test]
fn tiled_gemm_matches_reference_nonsquare_tile_aligned() {
    // 64x48x32: all divisible by 8 -> tiled simdgroup path; non-square so a
    // transposed or swapped lda/ldb would corrupt the result.
    let (got, want, used_fused) = run_matmul(64, 48, 32);
    assert!(used_fused, "expected the fused MulReduceSum matmul path");
    assert_close(&got, &want, "tiled 64x48x32");
}

#[test]
fn tiled_gemm_matches_reference_rectangular_k() {
    // Larger K (still /8), rectangular MxN.
    let (got, want, _) = run_matmul(24, 72, 128);
    assert_close(&got, &want, "tiled 24x72x128");
}

#[test]
fn matmul_fallback_non_tile_aligned_is_correct() {
    // 12x20x12: not divisible by 8 -> naive fallback. Must still be correct.
    let (got, want, _) = run_matmul(12, 20, 12);
    assert_close(&got, &want, "fallback 12x20x12");
}
