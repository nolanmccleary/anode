use eqtheory::arena::{allocate_arena, read_f32, upload};
use eqtheory::metal_backend::CuteMetal;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex};
use luminal::shape::IntExpr;
use luminal_metal::{CompileOptions, HostBuffer, MetalRuntime};
use metal::Buffer;
use std::time::Instant;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

struct BlockW {
    wq: Buffer,
    wk: Buffer,
    wv: Buffer,
    wo: Buffer,
    w1: Buffer,
    w2: Buffer,
}

fn alloc(be: &CuteMetal, keep: &mut Vec<Buffer>, n: usize) -> Buffer {
    let b = be.alloc(n);
    keep.push(b.clone());
    b
}

fn mha(
    be: &CuteMetal,
    cb: &metal::CommandBufferRef,
    keep: &mut Vec<Buffer>,
    x: &Buffer,
    w: &BlockW,
    s: usize,
    d: usize,
    h: usize,
    affine: bool,
) -> Buffer {
    let dh = d / h;
    let (di, si, dhi) = (d as i32, s as i32, dh as i32);
    let proj = [0, di, 1, 0];
    let q = alloc(be, keep, s * d);
    let k = alloc(be, keep, s * d);
    let v = alloc(be, keep, s * d);
    be.matmul_into(cb, x, &w.wq, &q, s, d, d, proj, proj, proj);
    be.matmul_into(cb, x, &w.wk, &k, s, d, d, proj, proj, proj);
    be.matmul_into(cb, x, &w.wv, &v, s, d, d, proj, proj, proj);

    let heads = [h as u32, s as u32, dh as u32, 0];
    let head_st = [0, dhi, di, 1];
    let (qh, kh, vh) = if affine {
        let qh = alloc(be, keep, h * s * dh);
        let kh = alloc(be, keep, h * s * dh);
        let vh = alloc(be, keep, h * s * dh);
        be.gather_into(cb, &q, &qh, heads, head_st);
        be.gather_into(cb, &k, &kh, heads, head_st);
        be.gather_into(cb, &v, &vh, heads, head_st);
        (Some(qh), Some(kh), Some(vh))
    } else {
        (None, None, None)
    };

    let scores = alloc(be, keep, h * s * s);
    for hh in 0..h {
        let (hhi, hsq, hsd) = (hh as i32, (hh * s * s) as i32, (hh * s * dh) as i32);
        let (ap, bp) = if affine {
            ([hsd, dhi, 1, 0], [hsd, 1, dhi, 0])
        } else {
            ([hhi * dhi, di, 1, 0], [hhi * dhi, 1, di, 0])
        };
        let a = if affine { qh.as_ref().unwrap() } else { &q };
        let b = if affine { kh.as_ref().unwrap() } else { &k };
        be.matmul_into(cb, a, b, &scores, s, dh, s, ap, bp, [hsq, si, 1, 0]);
    }
    be.softmax_into(cb, &scores, h * s, s);

    let merged = alloc(be, keep, s * d);
    for hh in 0..h {
        let (hhi, hsq, hsd) = (hh as i32, (hh * s * s) as i32, (hh * s * dh) as i32);
        let bp = if affine { [hsd, dhi, 1, 0] } else { [hhi * dhi, di, 1, 0] };
        let b = if affine { vh.as_ref().unwrap() } else { &v };
        be.matmul_into(cb, &scores, b, &merged, s, s, dh, [hsq, si, 1, 0], bp, [hhi * dhi, di, 1, 0]);
    }

    let out = alloc(be, keep, s * d);
    be.matmul_into(cb, &merged, &w.wo, &out, s, d, d, proj, proj, proj);
    out
}

fn block(
    be: &CuteMetal,
    cb: &metal::CommandBufferRef,
    keep: &mut Vec<Buffer>,
    x: &Buffer,
    w: &BlockW,
    s: usize,
    d: usize,
    h: usize,
    hidden: usize,
    affine: bool,
) -> Buffer {
    let (di, hid) = (d as i32, hidden as i32);
    let a = mha(be, cb, keep, x, w, s, d, h, affine);
    let y = alloc(be, keep, s * d);
    be.add_into(cb, x, &a, &y, s * d);
    let g = alloc(be, keep, s * hidden);
    be.matmul_into(cb, &y, &w.w1, &g, s, d, hidden, [0, di, 1, 0], [0, hid, 1, 0], [0, hid, 1, 0]);
    be.relu_into(cb, &g, s * hidden);
    let f = alloc(be, keep, s * d);
    be.matmul_into(cb, &g, &w.w2, &f, s, hidden, d, [0, hid, 1, 0], [0, di, 1, 0], [0, di, 1, 0]);
    let z = alloc(be, keep, s * d);
    be.add_into(cb, &y, &f, &z, s * d);
    z
}

fn forward(
    be: &CuteMetal,
    x_in: &Buffer,
    ws: &[BlockW],
    s: usize,
    d: usize,
    h: usize,
    hidden: usize,
    affine: bool,
) -> Vec<f32> {
    objc::rc::autoreleasepool(|| {
        let cb = be.command_buffer();
        let mut keep: Vec<Buffer> = Vec::new();
        let mut x = x_in.clone();
        for w in ws {
            x = block(be, cb, &mut keep, &x, w, s, d, h, hidden, affine);
        }
        cb.commit();
        cb.wait_until_completed();
        be.read(&x, s * d)
    })
}

struct Oracle {
    rt: MetalRuntime,
    arena: eqtheory::arena::Allocation,
    out: NodeIndex,
}

fn build_luminal(
    s: usize,
    d: usize,
    h: usize,
    hidden: usize,
    nblocks: usize,
    inputs: &[Vec<f32>],
) -> Oracle {
    let (si, di, hid, dhi) = (s as i64, d as i64, hidden as i64, (d / h) as i64);
    let mut g = Graph::new();
    let dim = IntExpr::from;
    let x = g.tensor([dim(si), dim(di)], DType::F32);
    let mut ids = vec![x.id];
    let mut cur = x;
    for _ in 0..nblocks {
        let wq = g.tensor([dim(di), dim(di)], DType::F32);
        let wk = g.tensor([dim(di), dim(di)], DType::F32);
        let wv = g.tensor([dim(di), dim(di)], DType::F32);
        let wo = g.tensor([dim(di), dim(di)], DType::F32);
        let w1 = g.tensor([dim(di), dim(hid)], DType::F32);
        let w2 = g.tensor([dim(hid), dim(di)], DType::F32);
        ids.extend([wq.id, wk.id, wv.id, wo.id, w1.id, w2.id]);
        let to_heads = |t: GraphTensor| t.split_dims(1, dim(dhi)).permute(&[1, 0, 2]);
        let qh = to_heads(cur.matmul(wq));
        let kh = to_heads(cur.matmul(wk));
        let vh = to_heads(cur.matmul(wv));
        let scores = qh.matmul(kh.transpose(1, 2));
        let p = scores.softmax(&[2]);
        let attn = p.matmul(vh);
        let merged = attn.permute(&[1, 0, 2]).merge_dims(1, 2);
        let ao = merged.matmul(wo);
        let y = cur + ao;
        let gg = y.matmul(w1).relu();
        let ff = gg.matmul(w2);
        cur = y + ff;
    }

    let mut rt = MetalRuntime::load(&g).unwrap();
    let search_inputs: FxHashMap<NodeIndex, HostBuffer> =
        ids.iter().zip(inputs).map(|(id, v)| (*id, v.clone().into())).collect();
    let opts = CompileOptions {
        generations: 3,
        generation_size: 4,
        trials: 2,
        seed: 0,
        search_log: false,
        ..CompileOptions::default()
    };
    rt.search(&Default::default(), &Default::default(), &search_inputs, &opts)
        .unwrap();
    let mut arena = allocate_arena(&rt).unwrap();
    for (id, v) in ids.iter().zip(inputs) {
        upload(&rt, &mut arena, *id, v.clone()).unwrap();
    }
    rt.execute(arena.buffer()).unwrap();
    Oracle { rt, arena, out: cur.id }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn time_min(iters: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e3
}

fn main() {
    let (s, d, h, hidden) = (128usize, 256usize, 8usize, 1024usize);
    println!("FAIR A/B (stacked transformer): cute vs affine on ONE shared kernel set");
    println!("S={s} D={d} H={h} FFN={hidden}, N blocks of MHA + residual + ReLU-FFN + residual");
    println!("cute reads head-reshapes by stride; affine materializes (gather3). same kernels.\n");
    println!(
        "{:>3} | {:>6} {:>6} {:>7} | {:>8} {:>8} | {:>9} {:>9} {:>7} | {:>6}",
        "N", "cuteK", "affK", "copies", "cute ms", "aff ms", "c-vs-a", "c-vs-lum", "fusion", "oracle"
    );
    println!("{}", "-".repeat(94));

    let be = CuteMetal::new();
    let mut all_ok = true;
    for nblocks in [1usize, 2, 4, 8] {
        let scaled = |n: usize, seed: u64| -> Vec<f32> { fill(n, seed).iter().map(|x| x * 0.05).collect() };
        let mut data = vec![fill(s * d, 1)];
        let mut seed = 2u64;
        for _ in 0..nblocks {
            for _ in 0..4 {
                data.push(scaled(d * d, seed));
                seed += 1;
            }
            data.push(scaled(d * hidden, seed));
            seed += 1;
            data.push(scaled(hidden * d, seed));
            seed += 1;
        }

        let x_buf = be.upload(&data[0]);
        let ws: Vec<BlockW> = (0..nblocks)
            .map(|b| {
                let o = 1 + 6 * b;
                BlockW {
                    wq: be.upload(&data[o]),
                    wk: be.upload(&data[o + 1]),
                    wv: be.upload(&data[o + 2]),
                    wo: be.upload(&data[o + 3]),
                    w1: be.upload(&data[o + 4]),
                    w2: be.upload(&data[o + 5]),
                }
            })
            .collect();

        let cute = forward(&be, &x_buf, &ws, s, d, h, hidden, false);
        let affine = forward(&be, &x_buf, &ws, s, d, h, hidden, true);
        let oracle = build_luminal(s, d, h, hidden, nblocks, &data);
        let lum = read_f32(&oracle.rt, &oracle.arena, oracle.out).unwrap();

        let peak = lum.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let c_vs_a = max_abs(&cute, &affine);
        let c_vs_lum = max_abs(&cute, &lum) / peak;
        let ok = c_vs_a < 1e-3 && c_vs_lum < 1e-4;
        all_ok &= ok;

        let cute_ms = time_min(15, || {
            std::hint::black_box(forward(&be, &x_buf, &ws, s, d, h, hidden, false));
        });
        let aff_ms = time_min(15, || {
            std::hint::black_box(forward(&be, &x_buf, &ws, s, d, h, hidden, true));
        });

        println!(
            "{:>3} | {:>6} {:>6} {:>7} | {:>8.3} {:>8.3} | {:>9.2e} {:>9.2e} {:>6.2}x | {:>6}",
            nblocks,
            nblocks * (2 * h + 10),
            nblocks * (2 * h + 13),
            3 * nblocks,
            cute_ms,
            aff_ms,
            c_vs_a,
            c_vs_lum,
            aff_ms / cute_ms,
            if ok { "match" } else { "FAIL" }
        );
    }
    println!("\ncopies = head-reshape materializations cute elides (3 per block)");
    println!("{}", if all_ok { "ALL MATCH" } else { "MISMATCH" });
    std::process::exit(if all_ok { 0 } else { 1 });
}
