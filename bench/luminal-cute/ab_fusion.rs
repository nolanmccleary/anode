//! A/B fusion harness (Step 2) — measures the value of GEMM epilogue fusion.
//!
//! Workload: D[m,n] = (A @ B^T)[m,n] + bias[m,n], F32. Both arms share the SAME
//! tiled simdgroup GEMM (Step 1), so the delta isolates FUSION, not kernel
//! quality:
//!   * UNFUSED: exclude the fused op -> plan is {tiled matmul kernel; add kernel}.
//!             The M*N matmul intermediate round-trips global memory.
//!   * FUSED:   exclude the standalone add -> plan is {MatmulBias kernel}. The
//!             bias is applied in the epilogue before the store; no round-trip.
//! Each arm's plan is pinned by the registry filter, so best_nanos is just the
//! min-over-trials GPU time of a fixed plan.
//!
//! Run: `cargo +1.99.1 run --release --example ab_fusion -p luminal_metal`
#![cfg(target_os = "macos")]

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal_metal::{CompileOptions, HostBuffer, MetalRuntime, metal_registry_filtered};
use rustc_hash::FxHashMap;

fn options() -> CompileOptions {
    CompileOptions {
        generations: 2,
        generation_size: 2,
        trials: 50,
        seed: 0,
        search_log: false,
        ..CompileOptions::default()
    }
}

fn fill(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % 17) as f32 - 8.0) * scale).collect()
}

/// Min `best_nanos` over several searches (min removes upward thermal/contention
/// noise — the right statistic for comparing peak kernel times), plus the
/// winning plan's compute-op labels. `fused=true` forces the fused MatmulBias
/// plan; `fused=false` forces the unfused {tiled matmul; add} plan.
fn measure(m: usize, n: usize, k: usize, fused: bool) -> (u128, Vec<String>) {
    const REPEATS: usize = 5;
    let mut best = u128::MAX;
    let mut ops = Vec::new();
    for _ in 0..REPEATS {
        // A search can fail to find a non-materializing plan (stochastic);
        // skip it rather than crash, and keep the min over the successes.
        if let Some((ns, got_ops)) = measure_once(m, n, k, fused) {
            if ns < best {
                best = ns;
            }
            ops = got_ops;
        }
    }
    (best, ops)
}

fn measure_once(m: usize, n: usize, k: usize, fused: bool) -> Option<(u128, Vec<String>)> {
    let mut g = Graph::new();
    let a = g.tensor(vec![m, k], DType::F32);
    let b = g.tensor(vec![n, k], DType::F32);
    let bias = g.tensor(vec![m, n], DType::F32);
    let _out = a.matmul(b.t()) + bias;

    // Pin BOTH arms to the tiled simdgroup GEMM mainloop by excluding the
    // decomposed reduction (`ReduceSumGeneric`) — then the matmul can only be
    // the tiled fused dot (`MulReduceSumGeneric`). This is the whole point of
    // Step 1: control for kernel quality so the delta is pure FUSION.
    //   * unfused: also exclude the fused op -> {tiled matmul; separate add}.
    //   * fused:   also exclude the standalone add -> {MatmulBias (tiled+bias)}.
    let other = if fused {
        "AddFunctionalGeneric"
    } else {
        "MatmulBiasGeneric"
    };
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| {
            row.label() != "ReduceSumGeneric" && row.label() != other
        }),
    )
    .unwrap();

    let inputs: FxHashMap<_, HostBuffer> = [
        (a.id, fill(m * k, 0.05).into()),
        (b.id, fill(n * k, 0.05).into()),
        (bias.id, fill(m * n, 0.1).into()),
    ]
    .into_iter()
    .collect();

    let outcome = rt
        .search(&Default::default(), &Default::default(), &inputs, &options())
        .ok()?;

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
    Some((outcome.best_nanos, ops))
}

fn main() {
    // (M, N, K). Square = compute-bound (big K); small-K = memory-bound, where
    // the eliminated M*N round-trip is a larger fraction of total traffic.
    // Products M*N*K kept under ~1 GB so no candidate materialization exceeds
    // the 4 GB arena budget. K descends: compute-bound -> memory-bound.
    let shapes = [
        (512usize, 512usize, 512usize), // compute-bound  (134M)
        (1024, 1024, 256),              // mixed          (268M)
        (2048, 2048, 64),               // memory-bound   (268M)
        (2048, 2048, 32),               // memory-bound   (134M)
    ];
    println!(
        "\nD = A@B^T + bias (F32). Both arms share the tiled simdgroup GEMM.\n\n\
         {:>16} | {:>13} | {:>13} | {:>9} | {:>8}",
        "M x N x K", "unfused (ms)", "fused (ms)", "speedup", "saved"
    );
    println!("{}", "-".repeat(76));

    for &(m, n, k) in &shapes {
        let (unfused_ns, unfused_ops) = measure(m, n, k, false);
        let (fused_ns, fused_ops) = measure(m, n, k, true);
        if unfused_ns == u128::MAX || fused_ns == u128::MAX {
            println!("{:>16} | (no executable plan found within arena budget)", format!("{m}x{n}x{k}"));
            continue;
        }
        // Both arms must use the tiled fused dot; only the epilogue differs.
        assert!(
            unfused_ops.iter().any(|o| o == "MulReduceSumGeneric")
                && unfused_ops.iter().any(|o| o == "AddFunctionalGeneric"),
            "unfused arm did not use tiled matmul + separate add: {unfused_ops:?}"
        );
        assert!(
            fused_ops.iter().any(|o| o == "MatmulBiasGeneric"),
            "fused arm did not use the fused op: {fused_ops:?}"
        );
        let unfused_ms = unfused_ns as f64 / 1e6;
        let fused_ms = fused_ns as f64 / 1e6;
        let speedup = unfused_ms / fused_ms;
        let saved = 100.0 * (unfused_ms - fused_ms) / unfused_ms;
        println!(
            "{:>16} | {:>13.4} | {:>13.4} | {:>8.2}x | {:>6.1}%",
            format!("{m}x{n}x{k}"),
            unfused_ms,
            fused_ms,
            speedup,
            saved
        );
        if (m, n, k) == shapes[0] {
            eprintln!("  unfused ops: {}", unfused_ops.join(", "));
            eprintln!("  fused ops:   {}", fused_ops.join(", "));
        }
    }

    println!("{}", "-".repeat(76));
    println!(
        "\nThe delta is the value of EPILOGUE FUSION: the unfused arm writes the\n\
         M*N matmul result to global memory and reads it back in the add kernel;\n\
         the fused arm applies bias in the GEMM epilogue, eliminating that\n\
         round-trip. Both arms use the identical tiled simdgroup GEMM mainloop.\n"
    );
}
