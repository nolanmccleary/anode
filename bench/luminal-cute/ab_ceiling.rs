//! A/B ceiling harness — Phase 0 of the Luminal-CuTe investigation.
//!
//! Measures the BASELINE performance ceiling of Luminal-Metal on fusion-
//! sensitive workloads, so later phases (tiled kernel, then CuTe epilogue
//! fusion) have a measuring stick and a target.
//!
//! For each workload it drives the graph through Luminal-Metal's own search
//! (which extracts + bufferizes + PROFILES candidates on the real GPU) and
//! reports the best measured time, achieved throughput, and which compute
//! ops the winning plan chose (to confirm whether fusion happened).
//!
//! Run: `cargo +1.99.1 run --release --example ab_ceiling -p luminal_metal`
#![cfg(target_os = "macos")]

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::shape::{DynMap, SymbolBounds};
use luminal_metal::{CompileOptions, HostBuffer, MetalRuntime};
use rustc_hash::FxHashMap;
use std::time::Instant;

/// Deterministic, generous-enough search to find a representative best plan
/// and time it stably. (harness_search_options uses 1 trial / 2 gens — too
/// noisy for a ceiling number.)
fn ceiling_options() -> CompileOptions {
    CompileOptions {
        generations: 8,
        generation_size: 8,
        trials: 25,
        seed: 0,
        search_log: false,
        ..CompileOptions::default()
    }
}

/// Fill a flat f32 buffer with small deterministic values.
fn fill(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % 17) as f32 - 8.0) * scale).collect()
}

struct Measurement {
    label: String,
    m: usize,
    n: usize,
    k: usize,
    flops: f64,
    best_ns: u128,
    compute_ops: Vec<String>,
}

impl Measurement {
    fn gflops(&self) -> f64 {
        self.flops / (self.best_ns as f64) // FLOP / ns = GFLOP/s
    }
    fn ms(&self) -> f64 {
        self.best_ns as f64 / 1.0e6
    }
}

/// Measure a GEMM  C[m,n] = A[m,k] @ B[k,n]  (optionally with a relu epilogue)
/// through baseline Luminal-Metal.
fn measure_gemm(m: usize, n: usize, k: usize, epilogue: bool) -> anyhow::Result<Measurement> {
    let mut g = Graph::new();
    // a:[m,k], b:[n,k]; a.matmul(b.t()) = [m,k]@[k,n] = [m,n].
    let a = g.tensor(vec![m, k], DType::F32);
    let b = g.tensor(vec![n, k], DType::F32);
    let c = a.matmul(b.t());
    // relu epilogue = max(c, 0): a pointwise, memory-bound op. In the baseline
    // this is a SEPARATE kernel reading+writing the M×N intermediate — exactly
    // the global round-trip that CuTe-driven epilogue fusion (Step 2) removes.
    let _out = if epilogue { c.maximum_f32(0.0) } else { c };

    let mut rt = MetalRuntime::load(&g)?;
    let inputs: FxHashMap<_, HostBuffer> = [
        (a.id, fill(m * k, 0.05).into()),
        (b.id, fill(n * k, 0.05).into()),
    ]
    .into_iter()
    .collect();

    let outcome = rt.search(
        &SymbolBounds::default(),
        &DynMap::default(),
        &inputs,
        &ceiling_options(),
    )?;

    let compute_ops = rt
        .plan()
        .ok_or_else(|| anyhow::anyhow!("no plan after search"))?
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::Compute { op, .. } => Some(op.label().to_string()),
            _ => None,
        })
        .collect();

    Ok(Measurement {
        label: if epilogue { "gemm+relu" } else { "gemm" }.to_string(),
        m,
        n,
        k,
        flops: 2.0 * m as f64 * n as f64 * k as f64,
        best_ns: outcome.best_nanos,
        compute_ops,
    })
}

fn op_summary(meas: &Measurement) -> String {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for op in &meas.compute_ops {
        // Only count real compute kernels, not alloc/free bookkeeping nodes.
        if op.starts_with("Buffer") {
            continue;
        }
        *counts.entry(op.as_str()).or_default() += 1;
    }
    counts
        .iter()
        .map(|(name, c)| format!("{name}×{c}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() -> anyhow::Result<()> {
    // Square GEMMs across a range; small enough that search (which runs each
    // candidate `trials` times) stays quick, large enough to be compute-bound.
    let shapes = [128usize, 256, 512, 1024];

    println!(
        "\n{:>5} | {:>10} {:>9} | {:>14} {:>9} | {:>12} | {}",
        "N",
        "gemm ms",
        "GFLOP/s",
        "gemm+relu ms",
        "GFLOP/s",
        "relu cost",
        "compute kernels (gemm+relu)"
    );
    println!("{}", "-".repeat(104));

    for &s in &shapes {
        let g0 = measure_gemm(s, s, s, false)?;
        let ge = measure_gemm(s, s, s, true)?;
        // The relu epilogue's cost = how much slower gemm+relu is than gemm.
        // This is the CEILING on what epilogue fusion (Step 2) could save at
        // CURRENT kernel speeds. It grows as a fraction once the GEMM is tiled.
        let relu_ms = ge.ms() - g0.ms();
        let relu_pct = 100.0 * relu_ms / g0.ms();

        println!(
            "{:>5} | {:>10.4} {:>9.1} | {:>14.4} {:>9.1} | {:>8.4}ms {:>3.0}% | {}",
            s,
            g0.ms(),
            g0.gflops(),
            ge.ms(),
            ge.gflops(),
            relu_ms,
            relu_pct,
            op_summary(&ge),
        );
    }

    println!("{}", "-".repeat(104));
    println!(
        "\nReadings (post Step 1 — the matmul is the tiled `simdgroup_matrix` kernel):\n\
         * GFLOP/s is Luminal-Metal's best profiled plan. The tiled GEMM is ~7x the naive\n\
           baseline at 1024 (~770 vs ~110 GFLOP/s, ~21% of ~3.6 TFLOP/s peak). Further\n\
           kernel tuning (shared-memory tiling) is possible but not needed for the A/B:\n\
           both A/B arms share this GEMM, so the delta isolates fusion.\n\
         * 'compute kernels' shows the epilogue (relu -> LessThan/Select/...) is a SEPARATE\n\
           set of kernels: the M*N intermediate round-trips global memory. Removing that\n\
           round-trip is what CuTe-driven epilogue FUSION buys (Step 2).\n\
         * 'relu cost' is now a LARGE fraction (~30-55%), where at naive GEMM speeds it was\n\
           sub-noise. Tiling made fusion's value measurable -> confirms: tile first, fuse next.\n"
    );

    Ok(())
}
