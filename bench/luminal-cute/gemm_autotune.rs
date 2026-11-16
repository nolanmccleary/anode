//! Tile-config as a search parameter — the payoff of mechanical construction.
//!
//! The GEMM kernel is generated ENTIRELY from a config (BM,BN,BK,TM,TN):
//! cute-layout's algebra rebuilds every layout (output thread-value partition,
//! A/B register fragments, tiles) from the config, and cute_msl lowers them
//! (nested crd2idx + slicing). So changing the config regenerates a correct
//! kernel for free. We then SEARCH the config space the way Luminal's search
//! does — profile every candidate on the real GPU, keep the fastest — and
//! report the autotuned best vs a fixed baseline.
//!
//! Run: cargo +1.99.1 run --release --example gemm_autotune -p luminal_metal
#![cfg(target_os = "macos")]

use cute_layout::algebra::mapping::composition;
use cute_layout::algebra::thread_value::make_layout_tv;
use cute_layout::algebra::tiling::zipped_divide;
use cute_layout::layout::Layout;
use cute_layout::tuple::IntTuple::{self, Leaf, Tuple};
use luminal_metal::cute_msl::{self, Coord};
use metal::{Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const M: i64 = 1024;
const N: i64 = 1024;
const K: i64 = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Cfg {
    bm: i64,
    bn: i64,
    bk: i64,
    tm: i64,
    tn: i64,
}

impl Cfg {
    fn threads(&self) -> i64 {
        (self.bm / self.tm) * (self.bn / self.tn)
    }
    fn smem_bytes(&self) -> i64 {
        (self.bm * self.bk + self.bn * self.bk) * 4 // plain shared layout
    }
    fn valid(&self) -> bool {
        self.bm % self.tm == 0
            && self.bn % self.tn == 0
            && M % self.bm == 0
            && N % self.bn == 0
            && K % self.bk == 0
            && self.threads() >= 32
            && self.threads() <= 1024
            && self.smem_bytes() <= 32768
    }
}

fn pair(a: i64, b: i64) -> IntTuple {
    Tuple(vec![Leaf(a), Leaf(b)])
}

/// The entire kernel source for one config, all addressing built by cute-layout
/// algebra and lowered by cute_msl.
fn kernel_source(c: Cfg) -> String {
    let Cfg { bm, bn, bk, tm, tn } = c;
    let tg = c.threads();

    // Output thread-value partition: make_layout_tv + composition.
    let thr = Layout::new(pair(bm / tm, bn / tn), pair(bn / tn, 1));
    let val = Layout::new(pair(tm, tn), pair(tn, 1));
    let (_, tv) = make_layout_tv(&thr, &val).unwrap();
    let out_tv = composition(&Layout::new(pair(bm, bn), pair(N, 1)), &tv).unwrap();
    let c_tile = cute_msl::layout_offset_tree(
        &out_tv,
        &Coord::Nest(vec![
            Coord::Nest(vec![Coord::leaf("tx"), Coord::leaf("ty")]),
            Coord::Nest(vec![Coord::leaf("j"), Coord::leaf("i")]),
        ]),
    )
    .unwrap();

    // A/B register fragments: zipped_divide + slice (per-thread base + loop free).
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

    // Global loads + shared stores (cute_msl-lowered from the layouts).
    let a_glob = cute_msl::layout_offset(&Layout::new(pair(M, K), pair(K, 1)), &["(row0 + r)", "(k0 + c)"]);
    let b_glob = cute_msl::layout_offset(&Layout::new(pair(N, K), pair(K, 1)), &["(col0 + r)", "(k0 + c)"]);
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
        bm = bm, bn = bn, bk = bk, tm = tm, tn = tn, tg = tg, k = K, n = N, bn_tn = bn / tn,
        as_base = as_base, as_free = as_free, bs_base = bs_base, bs_free = bs_free,
        as_store = as_store, bs_store = bs_store, a_glob = a_glob, b_glob = b_glob, c_tile = c_tile,
    )
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    println!("device: {}, GEMM {M}x{N}x{K}\n", device.name());
    let queue = device.new_command_queue();

    let a: Vec<f32> = (0..(M * K) as usize).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let b: Vec<f32> = (0..(N * K) as usize).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
    // CPU reference (one full GEMM; reused to check every candidate).
    let mut want = vec![0f32; (M * N) as usize];
    for m in 0..M as usize {
        for n in 0..N as usize {
            let mut acc = 0f64;
            for kk in 0..K as usize {
                acc += a[m * K as usize + kk] as f64 * b[n * K as usize + kk] as f64;
            }
            want[m * N as usize + n] = acc as f32;
        }
    }
    let a_buf = device.new_buffer_with_data(a.as_ptr() as *const _, (a.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let b_buf = device.new_buffer_with_data(b.as_ptr() as *const _, (b.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let c_buf = device.new_buffer((M * N * 4) as u64, MTLResourceOptions::StorageModeShared);
    let flop = 2.0 * M as f64 * N as f64 * K as f64;

    // The config search space.
    let mut configs = Vec::new();
    for bm in [32, 64, 128] {
        for bn in [32, 64, 128] {
            for bk in [8, 16, 32] {
                for tm in [4, 8] {
                    for tn in [4, 8] {
                        let c = Cfg { bm, bn, bk, tm, tn };
                        if c.valid() {
                            configs.push(c);
                        }
                    }
                }
            }
        }
    }

    let baseline = Cfg { bm: 64, bn: 64, bk: 8, tm: 4, tn: 4 };
    let mut best: Option<(f64, Cfg)> = None;
    let mut baseline_ms = 0.0;
    println!("searching {} valid configs (profile each, keep fastest)...\n", configs.len());

    for c in &configs {
        let src = kernel_source(*c);
        let lib = match device.new_library_with_source(&src, &metal::CompileOptions::new()) {
            Ok(l) => l,
            Err(_) => continue, // reject configs the backend can't compile
        };
        let pso = device.new_compute_pipeline_state_with_function(&lib.get_function("k", None).unwrap()).unwrap();
        if c.threads() as u64 > pso.max_total_threads_per_threadgroup() {
            continue;
        }
        let mut t_best = f64::MAX;
        for run in 0..24 {
            let t = Instant::now();
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_buffer(2, Some(&c_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new((N / c.bn) as u64, (M / c.bm) as u64, 1),
                MTLSize::new(c.threads() as u64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            if run >= 6 {
                t_best = t_best.min(t.elapsed().as_secs_f64());
            }
        }
        // Verify correctness before trusting the timing.
        let got = unsafe { std::slice::from_raw_parts(c_buf.contents() as *const f32, (M * N) as usize) };
        let max_err = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
        if max_err >= 1e-2 {
            println!("  {c:?}: WRONG (err {max_err:.1e}) — skipped");
            continue;
        }
        let gflops = flop / (t_best * 1e9);
        if *c == baseline {
            baseline_ms = t_best * 1e3;
        }
        if best.is_none_or(|(bt, _)| t_best < bt) {
            best = Some((t_best, *c));
            println!("  new best: {gflops:7.1} GFLOP/s  {c:?}");
        }
    }

    let (bt, bc) = best.expect("no config compiled+ran correctly");
    println!("\n--- autotuned best ---");
    println!("config:   {bc:?}");
    println!("time:     {:.4} ms   ({:.1} GFLOP/s)", bt * 1e3, flop / (bt * 1e9));
    if baseline_ms > 0.0 {
        println!(
            "baseline: {:.4} ms   ({:.1} GFLOP/s)   [BM64 BN64 BK8 TM4 TN4]",
            baseline_ms,
            flop / (baseline_ms * 1e6)
        );
        println!("speedup from autotuning the tile config: {:.2}x", baseline_ms / (bt * 1e3));
    }
    println!(
        "\nEvery candidate kernel was generated from its config by cute-layout\n\
         algebra + cute_msl; the search profiled each and kept the fastest. That\n\
         is CuTe-mechanical-construction with a Luminal-style profiling search on top."
    );
}
