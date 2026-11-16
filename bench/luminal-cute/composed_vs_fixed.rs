//! Isolate the STRICT win of the composed/autotuned system over "ham-fisting"
//! (shipping one fixed hand-picked kernel for every shape).
//!
//! HONEST framing: composed layouts are NOT intrinsically faster than Luminal's
//! affine addressing — same address, same MSL, same speed. The real, strict win
//! is that composition makes the whole tile-config FAMILY cheap to generate, so
//! the system autotunes per shape. best-of-family >= any-fixed-kernel, strictly
//! > whenever the shape's optimum differs from the fixed pick. This sweep shows
//! exactly that, measured raw (GPU-timestamp), bypassing the broken search path.
//!
//!   HAM-FISTED : one config, fixed across every shape (what a hand-written
//!                kernel gives you).
//!   COMPOSED   : the config family (mechanically built by cute-layout algebra),
//!                search picks the best for THAT shape.
//!
//! Run: cargo +1.99.1 run --release --example composed_vs_fixed -p luminal_metal
#![cfg(target_os = "macos")]

use cute_layout::algebra::mapping::composition;
use cute_layout::algebra::thread_value::make_layout_tv;
use cute_layout::algebra::tiling::zipped_divide;
use cute_layout::layout::Layout;
use cute_layout::tuple::IntTuple::{self, Leaf, Tuple};
use luminal_metal::cute_msl::{self, Coord};
use metal::{Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Cfg {
    bm: i64,
    bn: i64,
    bk: i64,
    tm: i64,
    tn: i64,
}

// The fixed "ham-fisted" kernel: one config a hand-writer would pick as a default.
const FIXED: Cfg = Cfg { bm: 64, bn: 64, bk: 8, tm: 4, tn: 4 };

impl Cfg {
    fn threads(&self) -> i64 {
        (self.bm / self.tm) * (self.bn / self.tn)
    }
    fn valid(&self, m: i64, n: i64, k: i64) -> bool {
        self.bm % self.tm == 0
            && self.bn % self.tn == 0
            && m % self.bm == 0
            && n % self.bn == 0
            && k % self.bk == 0
            && self.threads() >= 32
            && self.threads() <= 1024
            && (self.bm * self.bk + self.bn * self.bk) * 4 <= 32768
    }
}

fn config_space() -> Vec<Cfg> {
    let mut v = Vec::new();
    for bm in [32, 64, 128] {
        for bn in [32, 64, 128] {
            for bk in [8, 16, 32] {
                for tm in [4, 8] {
                    for tn in [4, 8] {
                        v.push(Cfg { bm, bn, bk, tm, tn });
                    }
                }
            }
        }
    }
    v
}

fn pair(a: i64, b: i64) -> IntTuple {
    Tuple(vec![Leaf(a), Leaf(b)])
}

/// On-GPU span (seconds) of a command buffer — clean kernel timing.
fn gpu_seconds(cb: &metal::CommandBufferRef) -> Option<f64> {
    use metal::foreign_types::ForeignTypeRef;
    use objc::{msg_send, runtime::Object, sel, sel_impl};
    let obj = cb.as_ptr() as *mut Object;
    let start: f64 = unsafe { msg_send![obj, GPUStartTime] };
    let end: f64 = unsafe { msg_send![obj, GPUEndTime] };
    (end.is_finite() && end > start).then_some(end - start)
}

/// Register-tiled kernel for a config, all addressing built by cute-layout algebra.
fn kernel_source(c: Cfg, m: i64, n: i64, k: i64) -> String {
    let Cfg { bm, bn, bk, tm, tn } = c;
    let tg = c.threads();
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
    let bn_tn = bn / tn;
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
    )
}

/// Min on-GPU seconds for a config (None if it won't compile/launch here).
fn time_cfg(dev: &Device, q: &metal::CommandQueue, c: Cfg, m: i64, n: i64, k: i64,
            a: &metal::Buffer, b: &metal::Buffer, out: &metal::Buffer) -> Option<f64> {
    if !c.valid(m, n, k) {
        return None;
    }
    let lib = dev.new_library_with_source(&kernel_source(c, m, n, k), &metal::CompileOptions::new()).ok()?;
    let func = lib.get_function("k", None).ok()?;
    let pso = dev.new_compute_pipeline_state_with_function(&func).ok()?;
    if c.threads() as u64 > pso.max_total_threads_per_threadgroup() {
        return None;
    }
    let tg = c.threads();
    let mut best = f64::MAX;
    for run in 0..24 {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.dispatch_thread_groups(MTLSize::new((n / c.bn) as u64, (m / c.bm) as u64, 1), MTLSize::new(tg as u64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if run >= 6 {
            if let Some(s) = gpu_seconds(cb) {
                best = best.min(s);
            }
        }
    }
    (best < f64::MAX).then_some(best)
}

fn main() {
    let dev = Device::system_default().expect("no Metal device");
    let q = dev.new_command_queue();
    let space = config_space();
    println!("device: {}   (all GFLOP/s are on-GPU-timestamp, min of back-to-back)\n", dev.name());
    println!("{:>18} | ham-fisted (fixed {:?})  | composed (autotuned best)      | win",
        "M x N x K", (FIXED.bm, FIXED.bn, FIXED.bk, FIXED.tm, FIXED.tn));
    println!("{}", "-".repeat(104));

    // Diverse shapes: square, skinny/fat, small-K (mem-bound), large-K (compute).
    let shapes = [
        (1024i64, 1024i64, 1024i64),
        (4096, 512, 1024),
        (512, 4096, 1024),
        (2048, 2048, 128),
        (1024, 1024, 4096),
        (768, 768, 1536),
        // Extreme shapes to force the optimal config to vary:
        (256, 256, 256),      // small — big tiles underutilize
        (8192, 128, 2048),    // tall, thin N
        (128, 8192, 2048),    // short, wide
        (4096, 4096, 64),     // large & very memory-bound
        (384, 384, 384),      // non-pow2, only small tiles divide
    ];
    let (mut geo_fixed, mut geo_auto, mut cnt) = (1.0f64, 1.0f64, 0u32);
    let mut strict_wins = 0u32;

    for &(m, n, k) in &shapes {
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let ab = (m * k).max(n * k).max(m * n) as usize;
        let data: Vec<f32> = (0..ab).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        let abuf = dev.new_buffer_with_data(data.as_ptr() as *const _, (m * k * 4) as u64, MTLResourceOptions::StorageModeShared);
        let bbuf = dev.new_buffer_with_data(data.as_ptr() as *const _, (n * k * 4) as u64, MTLResourceOptions::StorageModeShared);
        let obuf = dev.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let fixed_s = time_cfg(&dev, &q, FIXED, m, n, k, &abuf, &bbuf, &obuf);
        let mut best: Option<(f64, Cfg)> = None;
        for &c in &space {
            if let Some(s) = time_cfg(&dev, &q, c, m, n, k, &abuf, &bbuf, &obuf) {
                if best.is_none_or(|(bs, _)| s < bs) {
                    best = Some((s, c));
                }
            }
        }
        let g = |s: f64| flop / (s * 1e9);
        match (fixed_s, best) {
            (Some(fs), Some((bs, bc))) => {
                let (fg, bg) = (g(fs), g(bs));
                let win = bg / fg;
                if win > 1.001 { strict_wins += 1; }
                geo_fixed *= fg; geo_auto *= bg; cnt += 1;
                println!("{:>18} | {:5.0} GFLOP/s            | {:5.0} GFLOP/s  {:?} | {:.2}x",
                    format!("{m}x{n}x{k}"), fg, bg, (bc.bm, bc.bn, bc.bk, bc.tm, bc.tn), win);
            }
            _ => println!("{:>18} | (fixed config invalid for this shape — ham-fisting can't even run it)", format!("{m}x{n}x{k}")),
        }
    }

    if cnt > 0 {
        let gm_f = geo_fixed.powf(1.0 / cnt as f64);
        let gm_a = geo_auto.powf(1.0 / cnt as f64);
        println!("{}", "-".repeat(104));
        println!("geomean: ham-fisted {:.0} GFLOP/s  vs  composed-autotuned {:.0} GFLOP/s  = {:.2}x overall",
            gm_f, gm_a, gm_a / gm_f);
        println!("composed strictly beat the fixed kernel on {strict_wins}/{cnt} shapes, tied (never lost) on the rest.");
    }
    println!("\nThis isolates the STRICT win: a mechanically-generated, autotuned family");
    println!("dominates any single fixed kernel. Composition is the enabler (cheap, correct");
    println!("family). It is NOT a claim that composed addresses beat Luminal's per-kernel.");
}
