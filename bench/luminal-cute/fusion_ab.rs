//! Low-noise A/B benchmark for GEMM epilogue fusion (Step 2). Unlike the
//! search-based harness, this compiles each arm's plan ONCE (pinned by registry
//! filter), then times many back-to-back `execute()` calls and takes the min —
//! removing search stochasticity and most GPU jitter. Run explicitly:
//!   cargo +1.99.1 test -p luminal_metal --test fusion_ab -- --nocapture --ignored
#![cfg(target_os = "macos")]
mod support;

use luminal::bufferize::BufferNode;
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::{CompileOptions, MetalRuntime, metal_registry_filtered};
use std::time::Instant;
use support::TestTransfers;

fn opts() -> CompileOptions {
    // Enough exploration to reliably find the (forced) non-materializing plan.
    CompileOptions {
        generations: 6,
        generation_size: 8,
        trials: 3,
        seed: 0,
        search_log: false,
        ..CompileOptions::default()
    }
}

fn fill(n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % 17) as f32 - 8.0) * s).collect()
}

/// Compile the arm's plan (pinned by filter), then return the min per-execute
/// time in ms over many back-to-back runs, plus the plan's compute ops.
fn bench(m: usize, n: usize, k: usize, fused: bool) -> (f64, Vec<String>) {
    let mut g = Graph::new();
    let a = g.tensor(vec![m, k], DType::F32);
    let b = g.tensor(vec![n, k], DType::F32);
    let bias = g.tensor(vec![m, n], DType::F32);
    let out = a.matmul(b.t()) + bias;

    let other = if fused {
        "AddFunctionalGeneric"
    } else {
        "MatmulBiasGeneric"
    };
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|r| r.label() != "ReduceSumGeneric" && r.label() != other),
    )
    .unwrap();

    let av = fill(m * k, 0.05);
    let bv = fill(n * k, 0.05);
    let biasv = fill(m * n, 0.1);
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
        &opts(),
    )
    .unwrap();

    let mut ops: Vec<String> = rt
        .plan()
        .unwrap()
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::Compute { op, .. } if !op.label().starts_with("Buffer") => {
                Some(op.label().to_string())
            }
            _ => None,
        })
        .collect();
    ops.sort();

    let mut arena = support::allocate_arena(&rt).unwrap();
    rt.upload(&mut arena, a.id, av).unwrap();
    rt.upload(&mut arena, b.id, bv).unwrap();
    rt.upload(&mut arena, bias.id, biasv).unwrap();
    let _ = out;

    // Warmup, then min over batches (each batch amortizes CPU encode jitter).
    const WARMUP: usize = 20;
    const BATCHES: usize = 40;
    const PER_BATCH: usize = 10;
    for _ in 0..WARMUP {
        rt.execute(arena.buffer()).unwrap();
    }
    let mut best = f64::MAX;
    for _ in 0..BATCHES {
        let t = Instant::now();
        for _ in 0..PER_BATCH {
            rt.execute(arena.buffer()).unwrap();
        }
        best = best.min(t.elapsed().as_secs_f64() / PER_BATCH as f64);
    }
    (best * 1e3, ops)
}

#[test]
#[ignore = "benchmark; run with --ignored"]
fn fusion_ab_benchmark() {
    let shapes = [
        (512usize, 512usize, 512usize), // compute-bound
        (1024, 1024, 512),              // compute-bound
        (1024, 1024, 128),              // mixed
        (1024, 1024, 64),               // memory-bound
        (2048, 2048, 64),               // memory-bound
    ];
    eprintln!(
        "\nD = A@B^T + bias (F32), pinned-plan min-of-{}x{} timing.\n\n\
         {:>16} | {:>12} | {:>12} | {:>8} | {:>7}",
        40, 10, "M x N x K", "unfused ms", "fused ms", "speedup", "saved"
    );
    eprintln!("{}", "-".repeat(72));
    for &(m, n, k) in &shapes {
        let (uf, uops) = bench(m, n, k, false);
        let (f, fops) = bench(m, n, k, true);
        assert!(
            uops.iter().any(|o| o == "MulReduceSumGeneric")
                && uops.iter().any(|o| o == "AddFunctionalGeneric"),
            "{m}x{n}x{k} unfused used {uops:?}"
        );
        assert!(
            fops.iter().any(|o| o == "MatmulBiasGeneric"),
            "{m}x{n}x{k} fused used {fops:?}"
        );
        eprintln!(
            "{:>16} | {:>12.4} | {:>12.4} | {:>6.2}x | {:>5.1}%",
            format!("{m}x{n}x{k}"),
            uf,
            f,
            uf / f,
            100.0 * (uf - f) / uf
        );
    }
    eprintln!("{}", "-".repeat(72));
}
