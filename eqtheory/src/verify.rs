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

fn scores_via_cute(q: &[f32], k: &[f32], s: usize, d: usize) -> Vec<f32> {
    let kt = Modes::row_major(&[s as i64, d as i64])
        .apply(&Op::Permute(vec![1, 0]))
        .coalesce();
    let mut out = vec![0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            let mut acc = 0f32;
            for dd in 0..d {
                let addr = kt.crd2idx(&[dd as i64, j as i64]) as usize;
                acc += q[i * d + dd] * k[addr];
            }
            out[i * s + j] = acc;
        }
    }
    out
}

fn softmax_rows(scores: &[f32], s: usize) -> Vec<f32> {
    let mut out = vec![0f32; s * s];
    for i in 0..s {
        let row = &scores[i * s..(i + 1) * s];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for j in 0..s {
            let e = (row[j] - m).exp();
            out[i * s + j] = e;
            sum += e;
        }
        for j in 0..s {
            out[i * s + j] /= sum;
        }
    }
    out
}

fn cute_attention(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    s: usize,
    d: usize,
) -> Vec<f32> {
    let q = matmul(x, wq, s, d, d);
    let k = matmul(x, wk, s, d, d);
    let v = matmul(x, wv, s, d, d);
    let scores = scores_via_cute(&q, &k, s, d);
    let p = softmax_rows(&scores, s);
    let attn = matmul(&p, &v, s, s, d);
    matmul(&attn, wo, s, d, d)
}

struct Built {
    out: NodeIndex,
    inputs: Vec<(NodeIndex, Vec<f32>)>,
}

fn build_affine(g: &mut Graph, s: i64, d: i64) -> Built {
    let dim = IntExpr::from;
    let x = g.tensor([dim(s), dim(d)], DType::F32);
    let wq = g.tensor([dim(d), dim(d)], DType::F32);
    let wk = g.tensor([dim(d), dim(d)], DType::F32);
    let wv = g.tensor([dim(d), dim(d)], DType::F32);
    let wo = g.tensor([dim(d), dim(d)], DType::F32);
    let q = x.matmul(wq);
    let k = x.matmul(wk);
    let v = x.matmul(wv);
    let scores = q.matmul(k.transpose(0, 1));
    let p = scores.softmax(&[1]);
    let attn = p.matmul(v);
    let out: GraphTensor = attn.matmul(wo);
    let inputs = vec![
        (x.id, fill((s * d) as usize, 1)),
        (wq.id, fill((d * d) as usize, 2)),
        (wk.id, fill((d * d) as usize, 3)),
        (wv.id, fill((d * d) as usize, 4)),
        (wo.id, fill((d * d) as usize, 5)),
    ];
    Built { out: out.id, inputs }
}

fn main() {
    let (s, d) = (32usize, 64usize);
    println!("output-equality check: single-head attention  S={s}  D={d}\n");

    let mut g = Graph::new();
    let built = build_affine(&mut g, s as i64, d as i64);

    let mut rt = MetalRuntime::load(&g).unwrap();
    let search_inputs: FxHashMap<NodeIndex, HostBuffer> = built
        .inputs
        .iter()
        .map(|(id, v)| (*id, v.clone().into()))
        .collect();
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

    let g2 = |i: usize| &built.inputs[i].1;
    let cute = cute_attention(g2(0), g2(1), g2(2), g2(3), g2(4), s, d);

    assert_eq!(affine.len(), cute.len(), "output length mismatch");

    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut max_rel = 0f32;
    let mut peak = 0f32;
    for (a, c) in affine.iter().zip(cute.iter()) {
        let diff = (a - c).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff as f64;
        max_rel = max_rel.max(diff / (a.abs() + 1e-6));
        peak = peak.max(a.abs());
    }
    let mean_abs = sum_abs / affine.len() as f64;

    println!("Luminal (affine, real Metal plan) executed; cute-addressed reference executed.");
    println!("samples  affine[0..4] = {:?}", &affine[..4]);
    println!("         cute  [0..4] = {:?}", &cute[..4]);
    println!("\nelements        {}", affine.len());
    println!("peak |affine|   {peak:.5}");
    println!("max abs diff    {max_abs:.3e}");
    println!("mean abs diff   {mean_abs:.3e}");
    println!("max rel diff    {max_rel:.3e}");

    let ok = max_abs < 1e-2 && max_rel < 1e-2;
    println!(
        "\nverdict: {}  (fp reduction-order tolerance; GPU-tiled vs CPU-sequential)",
        if ok { "OUTPUTS MATCH" } else { "MISMATCH" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
