//! Correctness + selection of the fused matmul+bias op (Step 2). Forces the
//! fused path by excluding the standalone `AddFunctionalGeneric` kernel, so the
//! only producer for `(A @ B^T) + bias` is `MatmulBiasGeneric`.
#![cfg(target_os = "macos")]
mod support;

use luminal::bufferize::BufferNode;
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::{MetalRuntime, harness_search_options, metal_registry_filtered};
use support::TestTransfers;

fn reference(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            for ki in 0..k {
                acc += a[mi * k + ki] as f64 * b[ni * k + ki] as f64;
            }
            c[mi * n + ni] = acc as f32 + bias[mi * n + ni];
        }
    }
    c
}

fn run_fused(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>, bool) {
    let mut g = Graph::new();
    let a = g.tensor(vec![m, k], DType::F32);
    let b = g.tensor(vec![n, k], DType::F32);
    let bias = g.tensor(vec![m, n], DType::F32);
    let out = a.matmul(b.t()) + bias;

    let av: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let bv: Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let biasv: Vec<f32> = (0..m * n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let want = reference(m, n, k, &av, &bv, &biasv);

    // Exclude the standalone add kernel -> the only way to realize the epilogue
    // add is the fused MatmulBias op.
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| row.label() != "AddFunctionalGeneric"),
    )
    .unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &[
            (a.id, av.clone().into()),
            (b.id, bv.clone().into()),
            (bias.id, biasv.clone().into()),
        ]
        .into_iter()
        .collect(),
        &harness_search_options(),
    )
    .unwrap();

    let used_fused = rt.plan().unwrap().dag.node_weights().any(|node| {
        matches!(node, BufferNode::Compute { op, .. } if op.label() == "MatmulBiasGeneric")
    });

    let mut arena = support::allocate_arena(&rt).expect("allocate execution arena");
    rt.upload(&mut arena, a.id, av).unwrap();
    rt.upload(&mut arena, b.id, bv).unwrap();
    rt.upload(&mut arena, bias.id, biasv).unwrap();
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
fn fused_matmul_bias_tiled_matches_reference() {
    // Tile-aligned, non-square -> tiled fused simdgroup path.
    let (got, want, used_fused) = run_fused(64, 48, 32);
    assert!(used_fused, "expected the fused MatmulBias op to be selected");
    assert_close(&got, &want, "fused tiled 64x48x32");
}

#[test]
fn fused_matmul_bias_fallback_matches_reference() {
    // Non-tile-aligned -> per-element fused fallback (still one kernel).
    let (got, want, used_fused) = run_fused(12, 20, 12);
    assert!(used_fused, "expected the fused MatmulBias op to be selected");
    assert_close(&got, &want, "fused fallback 12x20x12");
}
