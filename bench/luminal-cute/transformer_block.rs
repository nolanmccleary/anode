//! Autotuning win on a real transformer block's GEMMs.
//!
//! Honest scope: this does NOT discover FlashAttention (that's an online-softmax
//! algorithmic rewrite, outside layout construction). It measures what the
//! composed/autotuned system DOES buy on a real model: every GEMM in a GPT-style
//! block gets its tile config autotuned (cute-layout builds each config's kernel),
//! vs one fixed hand-picked kernel for all of them. Reports per-GEMM and the
//! block-level speedup, weighted by where time is actually spent.
//!
//! Block (GPT-2-base-ish): S=512 tokens, D=768 hidden, H=12 heads, Dh=64,
//! FFN=3072. The GEMMs and their multiplicities:
//!   QKV proj   [S,D]x[D,3D]      x1
//!   attn QK^T  [S,Dh]x[Dh,S]     xH   (per head)
//!   attn @V    [S,S]x[S,Dh]      xH
//!   out proj   [S,D]x[D,D]       x1
//!   FFN up     [S,D]x[D,4D]      x1
//!   FFN down   [S,4D]x[4D,D]     x1
//!
//! Run: cargo +1.99.1 run --release --example transformer_block -p luminal_metal
#![cfg(target_os = "macos")]

use cute_layout::algebra::mapping::composition;
use cute_layout::algebra::thread_value::make_layout_tv;
use cute_layout::algebra::tiling::zipped_divide;
use cute_layout::layout::Layout;
use cute_layout::tuple::IntTuple::{self, Leaf, Tuple};
use luminal_metal::cute_msl::{self, Coord};
use metal::{Device, MTLResourceOptions, MTLSize};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Cfg { bm: i64, bn: i64, bk: i64, tm: i64, tn: i64 }

// The fixed "ham-fisted" kernel a hand-writer ships for every GEMM.
const FIXED: Cfg = Cfg { bm: 64, bn: 64, bk: 8, tm: 4, tn: 4 };

impl Cfg {
    fn threads(&self) -> i64 { (self.bm / self.tm) * (self.bn / self.tn) }
    fn valid(&self, m: i64, n: i64, k: i64) -> bool {
        self.bm % self.tm == 0 && self.bn % self.tn == 0
            && m % self.bm == 0 && n % self.bn == 0 && k % self.bk == 0
            && self.threads() >= 32 && self.threads() <= 1024
            && (self.bm * self.bk + self.bn * self.bk) * 4 <= 32768
    }
}

fn config_space() -> Vec<Cfg> {
    let mut v = Vec::new();
    for bm in [32, 64, 128] { for bn in [32, 64, 128] { for bk in [8, 16, 32] {
        for tm in [4, 8] { for tn in [4, 8] { v.push(Cfg { bm, bn, bk, tm, tn }); }}}}}
    v
}

fn pair(a: i64, b: i64) -> IntTuple { Tuple(vec![Leaf(a), Leaf(b)]) }

fn gpu_seconds(cb: &metal::CommandBufferRef) -> Option<f64> {
    use metal::foreign_types::ForeignTypeRef;
    use objc::{msg_send, runtime::Object, sel, sel_impl};
    let obj = cb.as_ptr() as *mut Object;
    let start: f64 = unsafe { msg_send![obj, GPUStartTime] };
    let end: f64 = unsafe { msg_send![obj, GPUEndTime] };
    (end.is_finite() && end > start).then_some(end - start)
}

fn kernel_source(c: Cfg, m: i64, n: i64, k: i64) -> String {
    let Cfg { bm, bn, bk, tm, tn } = c;
    let tg = c.threads();
    let thr = Layout::new(pair(bm / tm, bn / tn), pair(bn / tn, 1));
    let val = Layout::new(pair(tm, tn), pair(tn, 1));
    let (_, tv) = make_layout_tv(&thr, &val).unwrap();
    let out_tv = composition(&Layout::new(pair(bm, bn), pair(n, 1)), &tv).unwrap();
    let c_tile = cute_msl::layout_offset_tree(&out_tv, &Coord::Nest(vec![
        Coord::Nest(vec![Coord::leaf("tx"), Coord::leaf("ty")]),
        Coord::Nest(vec![Coord::leaf("j"), Coord::leaf("i")]),
    ])).unwrap();
    let frag = |rows: i64, reg: i64, thr_name: &str, reg_name: &str| {
        let f = zipped_divide(&Layout::new(pair(rows, bk), pair(bk, 1)), &[reg, bk]).unwrap();
        cute_msl::slice(&f, &Coord::Nest(vec![
            Coord::Nest(vec![Coord::leaf(reg_name), Coord::leaf("k")]),
            Coord::Nest(vec![Coord::fixed(thr_name), Coord::fixed("0")]),
        ])).unwrap()
    };
    let (as_base, as_free) = frag(bm, tm, "ty", "i");
    let (bs_base, bs_free) = frag(bn, tn, "tx", "j");
    let a_glob = cute_msl::layout_offset(&Layout::new(pair(m, k), pair(k, 1)), &["(row0 + r)", "(k0 + c)"]);
    let b_glob = cute_msl::layout_offset(&Layout::new(pair(n, k), pair(k, 1)), &["(col0 + r)", "(k0 + c)"]);
    let as_store = cute_msl::layout_offset(&Layout::new(pair(bm, bk), pair(bk, 1)), &["r", "c"]);
    let bs_store = cute_msl::layout_offset(&Layout::new(pair(bn, bk), pair(bk, 1)), &["r", "c"]);
    let bn_tn = bn / tn;
    format!(r#"#include <metal_stdlib>
using namespace metal;
kernel void k(device const float* A, device const float* B, device float* C,
              uint tid [[thread_index_in_threadgroup]], uint2 bid [[threadgroup_position_in_grid]]) {{
    threadgroup float As[{bm} * {bk}];
    threadgroup float Bs[{bn} * {bk}];
    uint tx = tid % {bn_tn}u; uint ty = tid / {bn_tn}u;
    threadgroup float* As_thr = As + {as_base};
    threadgroup float* Bs_thr = Bs + {bs_base};
    uint row0 = bid.y * {bm}u; uint col0 = bid.x * {bn}u;
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
}}"#)
}

fn time_cfg(dev: &Device, q: &metal::CommandQueue, c: Cfg, m: i64, n: i64, k: i64,
            a: &metal::Buffer, b: &metal::Buffer, out: &metal::Buffer) -> Option<f64> {
    if !c.valid(m, n, k) { return None; }
    let lib = dev.new_library_with_source(&kernel_source(c, m, n, k), &metal::CompileOptions::new()).ok()?;
    let func = lib.get_function("k", None).ok()?;
    let pso = dev.new_compute_pipeline_state_with_function(&func).ok()?;
    if c.threads() as u64 > pso.max_total_threads_per_threadgroup() { return None; }
    let tg = c.threads();
    let mut best = f64::MAX;
    for run in 0..24 {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(a), 0); enc.set_buffer(1, Some(b), 0); enc.set_buffer(2, Some(out), 0);
        enc.dispatch_thread_groups(MTLSize::new((n / c.bn) as u64, (m / c.bm) as u64, 1), MTLSize::new(tg as u64, 1, 1));
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        if run >= 6 { if let Some(s) = gpu_seconds(cb) { best = best.min(s); } }
    }
    (best < f64::MAX).then_some(best)
}

fn main() {
    let dev = Device::system_default().expect("no Metal device");
    let q = dev.new_command_queue();
    let space = config_space();

    // GPT-2-base-ish block (prefill, batch 1).
    let (s, d, h, dh, ff) = (512i64, 768i64, 12i64, 64i64, 3072i64);
    // (name, M, N, K, count)
    let gemms = [
        ("QKV proj",  s, 3 * d, d, 1),
        ("attn QK^T", s, s, dh, h),
        ("attn @V",   s, dh, s, h),
        ("out proj",  s, d, d, 1),
        ("FFN up",    s, ff, d, 1),
        ("FFN down",  s, d, ff, 1),
    ];
    println!("device: {}", dev.name());
    println!("block: S={s} D={d} heads={h} Dh={dh} FFN={ff}  (GPU-timestamp, min of back-to-back)\n");
    println!("{:>10} | shape (MxNxK)      xN | fixed us (xN) | autotuned us (xN)  best cfg      | win",
        "GEMM");
    println!("{}", "-".repeat(104));

    let (mut tot_fixed, mut tot_auto) = (0f64, 0f64);
    for &(name, m, n, k, count) in &gemms {
        let flop = 2.0 * m as f64 * n as f64 * k as f64 * count as f64;
        let ab = (m * k).max(n * k).max(m * n) as usize;
        let data: Vec<f32> = (0..ab).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        let abuf = dev.new_buffer_with_data(data.as_ptr() as *const _, (m * k * 4) as u64, MTLResourceOptions::StorageModeShared);
        let bbuf = dev.new_buffer_with_data(data.as_ptr() as *const _, (n * k * 4) as u64, MTLResourceOptions::StorageModeShared);
        let obuf = dev.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let fixed = time_cfg(&dev, &q, FIXED, m, n, k, &abuf, &bbuf, &obuf);
        let mut best: Option<(f64, Cfg)> = None;
        for &c in &space {
            if let Some(t) = time_cfg(&dev, &q, c, m, n, k, &abuf, &bbuf, &obuf) {
                if best.is_none_or(|(bt, _)| t < bt) { best = Some((t, c)); }
            }
        }
        if let (Some(ft), Some((bt, bc))) = (fixed, best) {
            let (fx, at) = (ft * count as f64, bt * count as f64); // total time for all `count` instances
            tot_fixed += fx; tot_auto += at;
            println!("{:>10} | {:>5}x{:<5}x{:<4} x{:<2} | {:7.1} us    | {:7.1} us  {:?} | {:.2}x  ({:.0} GF/s)",
                name, m, n, k, count, fx * 1e6, at * 1e6, (bc.bm, bc.bn, bc.bk, bc.tm, bc.tn),
                ft / bt, flop / (at * 1e9));
        }
    }
    println!("{}", "-".repeat(104));
    println!("BLOCK total GEMM time: ham-fisted {:.1} us  vs  autotuned {:.1} us  =  {:.2}x faster",
        tot_fixed * 1e6, tot_auto * 1e6, tot_fixed / tot_auto);
    println!("\nThis is the autotuning win the composed system buys on a real block's GEMMs —");
    println!("NOT FlashAttention (the N^2 scores still materialize; online softmax is a separate");
    println!("algorithmic rewrite a human must supply). Layout autotuning != algorithm discovery.");
}
