use eqtheory::arena::{allocate_arena, read_f32, upload};
use eqtheory::views::{Modes, Op};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex};
use luminal::shape::IntExpr;
use luminal_metal::{CompileOptions, HostBuffer, MetalRuntime};

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

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn softmax_last(x: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &mut x[i * cols..(i + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

fn cute_mha(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    s: usize,
    d: usize,
    h: usize,
) -> Vec<f32> {
    let dh = d / h;
    let q = matmul(x, wq, s, d, d);
    let k = matmul(x, wk, s, d, d);
    let v = matmul(x, wv, s, d, d);

    let head = Modes::row_major(&[s as i64, d as i64])
        .apply(&Op::Split(1, dh as i64))
        .apply(&Op::Permute(vec![1, 0, 2]));
    let addr = |hh: usize, ii: usize, dd: usize| head.crd2idx(&[hh as i64, ii as i64, dd as i64]) as usize;

    let mut scores = vec![0f32; h * s * s];
    for hh in 0..h {
        for i in 0..s {
            for j in 0..s {
                let mut acc = 0f32;
                for dd in 0..dh {
                    acc += q[addr(hh, i, dd)] * k[addr(hh, j, dd)];
                }
                scores[(hh * s + i) * s + j] = acc;
            }
        }
    }
    softmax_last(&mut scores, h * s, s);

    let mut attn = vec![0f32; h * s * dh];
    for hh in 0..h {
        for i in 0..s {
            for dd in 0..dh {
                let mut acc = 0f32;
                for j in 0..s {
                    acc += scores[(hh * s + i) * s + j] * v[addr(hh, j, dd)];
                }
                attn[(hh * s + i) * dh + dd] = acc;
            }
        }
    }

    let merge = Modes::row_major(&[h as i64, s as i64, dh as i64]).apply(&Op::Permute(vec![1, 0, 2]));
    let mut merged = vec![0f32; s * d];
    for i in 0..s {
        for hh in 0..h {
            for dd in 0..dh {
                let src = merge.crd2idx(&[i as i64, hh as i64, dd as i64]) as usize;
                merged[i * d + hh * dh + dd] = attn[src];
            }
        }
    }
    matmul(&merged, wo, s, d, d)
}

struct Built {
    out: NodeIndex,
    inputs: Vec<(NodeIndex, Vec<f32>)>,
}

fn build_mha(g: &mut Graph, s: i64, d: i64, h: i64) -> Built {
    let dim = IntExpr::from;
    let dh = d / h;
    let x = g.tensor([dim(s), dim(d)], DType::F32);
    let wq = g.tensor([dim(d), dim(d)], DType::F32);
    let wk = g.tensor([dim(d), dim(d)], DType::F32);
    let wv = g.tensor([dim(d), dim(d)], DType::F32);
    let wo = g.tensor([dim(d), dim(d)], DType::F32);

    let heads = |t: GraphTensor| t.split_dims(1, dim(dh)).permute(&[1, 0, 2]);
    let qh = heads(x.matmul(wq));
    let kh = heads(x.matmul(wk));
    let vh = heads(x.matmul(wv));

    let scores = qh.matmul(kh.transpose(1, 2));
    let p = scores.softmax(&[2]);
    let attn = p.matmul(vh);
    let merged = attn.permute(&[1, 0, 2]).merge_dims(1, 2);
    let out = merged.matmul(wo);

    let inputs = vec![
        (x.id, fill((s * d) as usize, 1)),
        (wq.id, fill((d * d) as usize, 2)),
        (wk.id, fill((d * d) as usize, 3)),
        (wv.id, fill((d * d) as usize, 4)),
        (wo.id, fill((d * d) as usize, 5)),
    ];
    Built { out: out.id, inputs }
}

fn run_case(s: usize, d: usize, h: usize) -> (f32, f32, f32, f32) {
    let mut g = Graph::new();
    let built = build_mha(&mut g, s as i64, d as i64, h as i64);
    let mut rt = MetalRuntime::load(&g).unwrap();
    let search_inputs: FxHashMap<NodeIndex, HostBuffer> =
        built.inputs.iter().map(|(id, v)| (*id, v.clone().into())).collect();
    let opts = CompileOptions {
        generations: 4,
        generation_size: 6,
        trials: 2,
        seed: 0,
        search_log: false,
        ..CompileOptions::default()
    };
    rt.search(&Default::default(), &Default::default(), &search_inputs, &opts)
        .unwrap();

    let mut arena = allocate_arena(&rt).unwrap();
    for (id, v) in &built.inputs {
        upload(&rt, &mut arena, *id, v.clone()).unwrap();
    }
    rt.execute(arena.buffer()).unwrap();
    let affine = read_f32(&rt, &arena, built.out).unwrap();

    let gi = |i: usize| &built.inputs[i].1;
    let cute = cute_mha(gi(0), gi(1), gi(2), gi(3), gi(4), s, d, h);
    assert_eq!(affine.len(), cute.len());

    let (mut max_abs, mut sum_abs, mut max_rel, mut peak) = (0f32, 0f64, 0f32, 0f32);
    for (a, c) in affine.iter().zip(cute.iter()) {
        let diff = (a - c).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff as f64;
        max_rel = max_rel.max(diff / (a.abs() + 1e-6));
        peak = peak.max(a.abs());
    }
    (peak, max_abs, (sum_abs / affine.len() as f64) as f32, max_rel)
}

fn main() {
    println!("multi-head attention — Luminal (Metal) vs cute-addressed reference");
    println!("split(D->H,Dh) + permute + transpose + merge, all addressed through cute\n");
    println!(
        "{:>5} {:>5} {:>4} {:>8} | {:>10} {:>11} {:>11} {:>11} | {:>7}",
        "S", "D", "H", "elems", "peak|out|", "max abs", "mean abs", "max rel", "verdict"
    );
    println!("{}", "-".repeat(86));

    let cases = [
        (32usize, 64usize, 4usize),
        (64, 128, 8),
        (128, 256, 8),
        (256, 512, 8),
        (512, 512, 16),
        (512, 1024, 16),
    ];
    let mut all_ok = true;
    for (s, d, h) in cases {
        let (peak, max_abs, mean_abs, max_rel) = run_case(s, d, h);
        let ok = max_rel < 5e-3 || max_abs < 1e-3;
        all_ok &= ok;
        println!(
            "{:>5} {:>5} {:>4} {:>8} | {:>10.4} {:>11.3e} {:>11.3e} {:>11.3e} | {:>7}",
            s,
            d,
            h,
            s * d,
            peak,
            max_abs,
            mean_abs,
            max_rel,
            if ok { "MATCH" } else { "MISMATCH" }
        );
    }
    println!(
        "\n{}",
        if all_ok { "ALL CASES MATCH" } else { "SOME CASES MISMATCH" }
    );
    std::process::exit(if all_ok { 0 } else { 1 });
}
