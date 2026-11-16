//! How well does config-in-the-e-graph-search actually work?
//!
//! Two measurements per GEMM size:
//!   (A) RAW: profile each of the 4 MATMUL_CONFIGS' kernels directly on the GPU
//!       (the ground-truth per-config spread + the true fastest).
//!   (B) INTEGRATED: run Luminal's genome search with only the MatmulTiled config
//!       variants available, read the delivered GPU time (`outcome.best_nanos`)
//!       and the config the genome actually selected (downcast the plan op).
//!
//! Then report: did the integrated search pick the fastest config, and what is
//! the autotune win (selected vs the fixed config-0 baseline)?
//!
//! Run: cargo +1.99.1 run --release --example config_search_bench -p luminal_metal
#![cfg(target_os = "macos")]

use cute_layout::algebra::mapping::composition;
use cute_layout::algebra::thread_value::make_layout_tv;
use cute_layout::algebra::tiling::zipped_divide;
use cute_layout::layout::Layout;
use cute_layout::tuple::IntTuple::{self, Leaf, Tuple};
use luminal::bufferize::BufferNode;
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::cute_msl::{self, Coord};
use luminal_metal::ops::matmul_tiled::MatmulTiledDps;
use luminal_metal::{CompileOptions, MetalRuntime, metal_registry_filtered};
use metal::{Device, MTLResourceOptions, MTLSize};
use rustc_hash::FxHashMap;
use std::time::Instant;

// Must match kernels.rs::MATMUL_CONFIGS (the configs the e-graph mints).
const CONFIGS: &[(i64, i64, i64, i64, i64)] = &[
    (64, 64, 8, 4, 4),
    (128, 64, 8, 4, 8),
    (64, 128, 8, 8, 4),
    (128, 128, 16, 8, 8),
];

fn pair(a: i64, b: i64) -> IntTuple {
    Tuple(vec![Leaf(a), Leaf(b)])
}

/// The exact cute-constructed register-tiled kernel (same addressing the
/// integrated `matmul_tiled_config_kernel` emits; only the ABI arg list differs).
fn kernel_source(c: (i64, i64, i64, i64, i64), m: i64, n: i64, k: i64) -> String {
    let (bm, bn, bk, tm, tn) = c;
    let tg = (bm / tm) * (bn / tn);
    let thr = Layout::new(pair(bm / tm, bn / tn), pair(bn / tn, 1));
    let val = Layout::new(pair(tm, tn), pair(tn, 1));
    let (_, tv) = make_layout_tv(&thr, &val).unwrap();
    let out_tv = composition(&Layout::new(pair(bm, bn), pair(n, 1)), &tv).unwrap();
    let c_tile = cute_msl::layout_offset_tree(
        &out_tv,
        &Coord::Nest(vec![
            Coord::Nest(vec![Coord::leaf("tx"), Coord::leaf("ty")]),
            Coord::Nest(vec![Coord::leaf("j"), Coord::leaf("i")]),
        ]),
    )
    .unwrap();
    let frag = |rows: i64, reg: i64, thr_name: &str, reg_name: &str| {
        let f = zipped_divide(&Layout::new(pair(rows, bk), pair(bk, 1)), &[reg, bk]).unwrap();
        cute_msl::slice(
            &f,
            &Coord::Nest(vec![
                Coord::Nest(vec![Coord::leaf(reg_name), Coord::leaf("k")]),
                Coord::Nest(vec![Coord::fixed(thr_name), Coord::fixed("0")]),
            ]),
        )
        .unwrap()
    };
    let (as_base, as_free) = frag(bm, tm, "ty", "i");
    let (bs_base, bs_free) = frag(bn, tn, "tx", "j");
    let a_glob = cute_msl::layout_offset(&Layout::new(pair(m, k), pair(k, 1)), &["(row0 + r)", "(k0 + c)"]);
    let b_glob = cute_msl::layout_offset(&Layout::new(pair(n, k), pair(k, 1)), &["(col0 + r)", "(k0 + c)"]);
    let as_store = cute_msl::layout_offset(&Layout::new(pair(bm, bk), pair(bk, 1)), &["r", "c"]);
    let bs_store = cute_msl::layout_offset(&Layout::new(pair(bn, bk), pair(bk, 1)), &["r", "c"]);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void k(device const float* A, device const float* B, device float* C,
              uint tid [[thread_index_in_threadgroup]],
              uint2 bid [[threadgroup_position_in_grid]]) {{
    threadgroup float As[{bm} * {bk}];
    threadgroup float Bs[{bn} * {bk}];
    uint tx = tid % {bn_tn}u;
    uint ty = tid / {bn_tn}u;
    threadgroup float* As_thr = As + {as_base};
    threadgroup float* Bs_thr = Bs + {bs_base};
    uint row0 = bid.y * {bm}u;
    uint col0 = bid.x * {bn}u;
    float acc[{tm}][{tn}];
    for (uint i = 0u; i < {tm}u; ++i) for (uint j = 0u; j < {tn}u; ++j) acc[i][j] = 0.0f;
    for (uint k0 = 0u; k0 < {k}u; k0 += {bk}u) {{
        for (uint t = tid; t < {bm}u * {bk}u; t += {tg}u) {{ uint r = t / {bk}u, c = t % {bk}u; As[{as_store}] = A[{a_glob}]; }}
        for (uint t = tid; t < {bn}u * {bk}u; t += {tg}u) {{ uint r = t / {bk}u, c = t % {bk}u; Bs[{bs_store}] = B[{b_glob}]; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < {bk}u; ++k) {{
            float a_reg[{tm}], b_reg[{tn}];
            for (uint i = 0u; i < {tm}u; ++i) a_reg[i] = As_thr[{as_free}];
            for (uint j = 0u; j < {tn}u; ++j) b_reg[j] = Bs_thr[{bs_free}];
            for (uint i = 0u; i < {tm}u; ++i) for (uint j = 0u; j < {tn}u; ++j) acc[i][j] += a_reg[i] * b_reg[j];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    device float* Ctile = C + (ulong)row0 * {n}u + col0;
    for (uint i = 0u; i < {tm}u; ++i) for (uint j = 0u; j < {tn}u; ++j) Ctile[{c_tile}] = acc[i][j];
}}"#,
        bn_tn = bn / tn,
    )
}

fn valid(c: (i64, i64, i64, i64, i64), m: i64, n: i64, k: i64) -> bool {
    let (bm, bn, bk, tm, tn) = c;
    m % bm == 0 && n % bn == 0 && k % bk == 0 && bm % tm == 0 && bn % tn == 0
}

/// Raw per-config GPU time (seconds, min of runs). Returns None if invalid.
fn raw_time(dev: &Device, q: &metal::CommandQueue, a: &metal::Buffer, b: &metal::Buffer,
            cbuf: &metal::Buffer, c: (i64, i64, i64, i64, i64), m: i64, n: i64, k: i64) -> Option<f64> {
    if !valid(c, m, n, k) {
        return None;
    }
    let (bm, bn, _, tm, tn) = c;
    let tg = (bm / tm) * (bn / tn);
    let lib = dev.new_library_with_source(&kernel_source(c, m, n, k), &metal::CompileOptions::new()).ok()?;
    let func = lib.get_function("k", None).ok()?;
    let pso = dev.new_compute_pipeline_state_with_function(&func).ok()?;
    if tg as u64 > pso.max_total_threads_per_threadgroup() {
        return None;
    }
    let mut best = f64::MAX;
    for run in 0..24 {
        let t = Instant::now();
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(cbuf), 0);
        enc.dispatch_thread_groups(MTLSize::new((n / bn) as u64, (m / bm) as u64, 1),
                                   MTLSize::new(tg as u64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if run >= 6 {
            best = best.min(t.elapsed().as_secs_f64());
        }
    }
    Some(best)
}

fn integrated(m: usize, n: usize, k: usize, steady: bool, gpu_ts: bool) -> Option<(f64, i64)> {
    let mut g = Graph::new();
    let a = g.tensor(vec![m, k], DType::F32);
    let b = g.tensor(vec![n, k], DType::F32);
    let _out = a.matmul(b.t());
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| {
            row.label() != "MulReduceSumGeneric" && row.label() != "ReduceSumGeneric"
        }),
    )
    .ok()?;
    let av: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
    let bv: Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
    let inputs: FxHashMap<_, _> = [(a.id, av.into()), (b.id, bv.into())].into_iter().collect();
    // Min best_nanos over several searches with DIFFERENT seeds (diverse
    // exploration) + many trials (low per-candidate noise) — the best shot at
    // the genome actually selecting the fastest config.
    let mut best_ns = f64::MAX;
    let mut cfg = -1;
    for s in 0..2u64 {
        let opts = CompileOptions {
            generations: 12,
            generation_size: 12,
            trials: 8,
            seed: s,
            steady_state_profiling: steady,
            gpu_timestamp_profiling: gpu_ts,
            search_log: false,
            ..CompileOptions::default()
        };
        let outcome = rt.search(&Default::default(), &Default::default(), &inputs, &opts).ok()?;
        if (outcome.best_nanos as f64) < best_ns {
            best_ns = outcome.best_nanos as f64;
            cfg = rt.plan().unwrap().dag.node_weights().find_map(|node| match node {
                BufferNode::Compute { op, .. } => op.as_any().downcast_ref::<MatmulTiledDps>().map(|o| o.config),
                _ => None,
            }).unwrap_or(-1);
        }
    }
    Some((best_ns / 1e9, cfg))
}

fn main() {
    let dev = Device::system_default().expect("no Metal device");
    let q = dev.new_command_queue();
    println!("device: {}\n", dev.name());
    println!("{:>14} | raw per-config GFLOP/s (0..3)         | raw best  | cfg0 base | INTEGRATED search      | picked best?",
        "M x N x K");
    println!("{}", "-".repeat(118));

    for &(m, n, k) in &[(1024usize, 1024, 1024), (2048, 2048, 2048)] {
        let (mi, ni, ki) = (m as i64, n as i64, k as i64);
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let ab = (m * k).max(n * k);
        let a: Vec<f32> = (0..ab).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        let abuf = dev.new_buffer_with_data(a.as_ptr() as *const _, (m * k * 4) as u64, MTLResourceOptions::StorageModeShared);
        let bbuf = dev.new_buffer_with_data(a.as_ptr() as *const _, (n * k * 4) as u64, MTLResourceOptions::StorageModeShared);
        let cbuf = dev.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

        // (A) raw per-config.
        let mut raw: Vec<Option<f64>> = Vec::new();
        for &c in CONFIGS {
            raw.push(raw_time(&dev, &q, &abuf, &bbuf, &cbuf, c, mi, ni, ki));
        }
        let gfs: Vec<String> = raw.iter().map(|t| match t {
            Some(t) => format!("{:5.0}", flop / (t * 1e9)),
            None => "  n/a".to_string(),
        }).collect();
        let (best_idx, best_t) = raw.iter().enumerate()
            .filter_map(|(i, t)| t.map(|t| (i, t)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        let best_g = flop / (best_t * 1e9);
        let cfg0_g = raw[0].map(|t| flop / (t * 1e9)).unwrap_or(0.0);

        // (B) integrated search: default (mean wall-clock) vs steady-state
        // (min wall-clock) vs GPU-timestamp (min on-GPU time) profiler.
        let mark = |cfg: i64| if cfg == best_idx as i64 { "YES" } else if cfg < 0 { "?" } else { "no" };
        let (def_s, def_cfg) = integrated(m, n, k, false, false).unwrap_or((f64::NAN, -2));
        let (st_s, st_cfg) = integrated(m, n, k, true, false).unwrap_or((f64::NAN, -2));
        let (gp_s, gp_cfg) = integrated(m, n, k, false, true).unwrap_or((f64::NAN, -2));

        println!("{:>14} | [{}] | best {:4.0}(#{}) cfg0 {:4.0} | default {:4.0}(#{} {}) | steady {:4.0}(#{} {}) | gpu-ts {:4.0}(#{} {})",
            format!("{m}x{n}x{k}"), gfs.join(" "), best_g, best_idx, cfg0_g,
            flop / (def_s * 1e9), def_cfg, mark(def_cfg),
            flop / (st_s * 1e9), st_cfg, mark(st_cfg),
            flop / (gp_s * 1e9), gp_cfg, mark(gp_cfg));
    }

    println!("\nraw [#0..#3] = each config timed directly on the GPU (min of back-to-back).");
    println!("best = fastest config (raw); cfg0 = fixed baseline config #0.");
    println!("default = genome search w/ Luminal's mean/single-shot WALL-CLOCK profiler.");
    println!("steady  = min-of-N back-to-back WALL-CLOCK profiler.");
    println!("gpu-ts  = min-of-N back-to-back ON-GPU-TIME profiler (GPUStartTime/EndTime).");
    println!("#k = config the genome picked; YES/no = did it pick the raw-fastest config.");

    // A single GEMM is the WORST case for the profiler: one short kernel per
    // execute(), so per-execute overhead + clock-ramp gaps dominate. A realistic
    // workload runs MANY kernels back-to-back in one execute, keeping the GPU
    // continuously busy. Test whether that restores the measured throughput.
    println!("\n-- matmul chain: R back-to-back GEMMs in one plan (realistic multi-kernel) --");
    println!("(raw single-1024^3 ceiling ~713; isolated search delivered ~560-610 above)");
    for &(nn, reps) in &[(1024usize, 1usize), (1024, 4), (1024, 16)] {
        let flop = reps as f64 * 2.0 * (nn as f64).powi(3);
        let def = chain(nn, reps, false).map(|s| flop / (s * 1e9)).unwrap_or(f64::NAN);
        let st = chain(nn, reps, true).map(|s| flop / (s * 1e9)).unwrap_or(f64::NAN);
        println!("  {nn}^3 x{reps:<2} kernels/execute: default {def:4.0} | steady {st:4.0} GFLOP/s (aggregate)");
    }
    println!("If throughput rises with chain length, the profiler's fidelity gap is a");
    println!("single-kernel artifact that realistic (multi-op) workloads largely escape.");
}

/// A chain of `reps` back-to-back GEMMs (x <- x @ wᵀ), so one execute() keeps the
/// GPU continuously busy — the multi-kernel regime a real model runs in. Returns
/// the min best_nanos (seconds) over a couple searches.
fn chain(n: usize, reps: usize, steady: bool) -> Option<f64> {
    let mut g = Graph::new();
    let x0 = g.tensor(vec![n, n], DType::F32);
    let w = g.tensor(vec![n, n], DType::F32);
    let mut x = x0;
    for _ in 0..reps {
        x = x.matmul(w.t());
    }
    let _out = x;
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| {
            row.label() != "MulReduceSumGeneric" && row.label() != "ReduceSumGeneric"
        }),
    )
    .ok()?;
    // Small magnitudes keep the chained values finite; GEMM timing is data-independent.
    let xv: Vec<f32> = vec![0.001; n * n];
    let wv: Vec<f32> = vec![0.001; n * n];
    let inputs: FxHashMap<_, _> = [(x0.id, xv.into()), (w.id, wv.into())].into_iter().collect();
    let mut best_ns = f64::MAX;
    for s in 0..2u64 {
        let opts = CompileOptions {
            generations: 12,
            generation_size: 12,
            trials: 8,
            seed: s,
            steady_state_profiling: steady,
            search_log: false,
            ..CompileOptions::default()
        };
        if let Ok(outcome) = rt.search(&Default::default(), &Default::default(), &inputs, &opts) {
            best_ns = best_ns.min(outcome.best_nanos as f64);
        }
    }
    Some(best_ns / 1e9)
}
