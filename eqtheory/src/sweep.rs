use eqtheory::views::{
    apply_view, cute_lower, cute_verified, deep_round_trip, preamble, Model, Op,
};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::NodeIndex;
use luminal::shape::IntExpr;
use std::time::Instant;

fn block_chains(s: i64, d: i64, dh: i64) -> Vec<Model> {
    let qkv = |tag: &str| Model {
        name: format!("{tag} head-reshape"),
        dims: vec![s, d],
        ops: vec![Op::Split(1, dh), Op::Permute(vec![1, 0, 2])],
    };
    let out = Model {
        name: "out merge".into(),
        dims: vec![s, d],
        ops: vec![
            Op::Split(1, dh),
            Op::Permute(vec![1, 0, 2]),
            Op::Permute(vec![1, 0, 2]),
            Op::Merge(1, 2),
        ],
    };
    vec![qkv("q"), qkv("k"), qkv("v"), out]
}

fn repeated(blocks: usize, s: i64, d: i64, dh: i64) -> Vec<Model> {
    let mut chains = Vec::new();
    for _ in 0..blocks {
        chains.extend(block_chains(s, d, dh));
    }
    chains
}

struct Point {
    name: String,
    chains: Vec<Model>,
}

fn build_all(chains: &[Model]) -> (Graph, Vec<NodeIndex>) {
    let mut g = Graph::new();
    let ids = chains
        .iter()
        .map(|m| {
            let shape: Vec<IntExpr> = m.dims.iter().map(|&d| IntExpr::from(d)).collect();
            let mut v = g.tensor(shape.as_slice(), DType::F32).view();
            for op in &m.ops {
                v = apply_view(v, op);
            }
            v.finish().id
        })
        .collect();
    (g, ids)
}

struct Affine {
    total_ms: f64,
    construct_ms: f64,
    saturate_ms: f64,
    rows: usize,
    nodes: usize,
}

fn affine_latency(pre: &str, chains: &[Model]) -> Affine {
    let mut best = f64::INFINITY;
    let mut out = Affine { total_ms: 0.0, construct_ms: 0.0, saturate_ms: 0.0, rows: 0, nodes: 0 };
    for _ in 0..3 {
        let c0 = Instant::now();
        let (g, ids) = build_all(chains);
        let seed = g.logical.render(&ids).unwrap();
        let construct = c0.elapsed().as_secs_f64();

        let s0 = Instant::now();
        let mut eg = luminal::egglog_snippet::new_egraph();
        eg.parse_and_run_program(None, pre).unwrap();
        eg.parse_and_run_program(None, &seed).unwrap();
        eg.parse_and_run_program(None, "(run-schedule (saturate (run prop)))").unwrap();
        for _ in 0..16 {
            eg.parse_and_run_program(
                None,
                "(run-schedule (run main_ruleset) (run prop) (run subst-walk))",
            )
            .unwrap();
        }
        let saturate = s0.elapsed().as_secs_f64();
        let total = construct + saturate;
        if total < best {
            best = total;
            out = Affine {
                total_ms: total * 1e3,
                construct_ms: construct * 1e3,
                saturate_ms: saturate * 1e3,
                rows: eg.num_tuples(),
                nodes: ["IntAdd", "IntMul", "IntTruncDiv", "IntTruncRem", "CoordVar", "IntLit"]
                    .iter()
                    .map(|t| seed.matches(t).count())
                    .sum(),
            };
        }
    }
    out
}

fn cute_latency(chains: &[Model]) -> (f64, usize) {
    let reps = 2000usize;
    let mut best = f64::INFINITY;
    let mut nodes = 0usize;
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..reps {
            for m in chains {
                let lowered = cute_lower(m);
                std::hint::black_box(lowered.nodes());
            }
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        if dt < best {
            best = dt;
            nodes = chains.iter().map(|m| cute_lower(m).nodes()).sum();
        }
    }
    (best * 1e3, nodes)
}

fn main() {
    let (s, d, dh) = (128i64, 256i64, 64i64);
    let points = vec![
        Point { name: "MLP".into(), chains: vec![] },
        Point { name: "Attention (1 blk)".into(), chains: repeated(1, s, d, dh) },
        Point { name: "GPT-2 (2 blk)".into(), chains: repeated(2, s, d, dh) },
        Point { name: "GPT-2 (4 blk)".into(), chains: repeated(4, s, d, dh) },
        Point { name: "GPT-2 (8 blk)".into(), chains: repeated(8, s, d, dh) },
        Point { name: "GPT-2 (12 blk)".into(), chains: repeated(12, s, d, dh) },
        Point { name: "deep-view x8".into(), chains: vec![deep_round_trip(vec![s, d], dh, 8)] },
        Point { name: "deep-view x12".into(), chains: vec![deep_round_trip(vec![s, d], dh, 12)] },
        Point { name: "deep-view x16".into(), chains: vec![deep_round_trip(vec![s, d], dh, 16)] },
        Point { name: "deep-view x20".into(), chains: vec![deep_round_trip(vec![s, d], dh, 20)] },
    ];

    let pre = preamble();
    println!("compile latency sweep: layout resolution, affine (Luminal) vs cute\n");
    println!(
        "{:>3}  {:<18} {:>6} | {:>10} {:>9} {:>9} {:>8} {:>7} | {:>10} {:>6} | {:>5}",
        "x", "model", "chains", "aff-tot", "construct", "saturate", "seed-nd", "rows", "cute-ms", "cute", "same"
    );
    println!("{}", "-".repeat(108));

    let mut csv = String::from(
        "x,model,chains,affine_ms,construct_ms,saturate_ms,seed_nodes,egraph_rows,cute_ms,cute_nodes\n",
    );
    for (x, p) in points.iter().enumerate() {
        let a = affine_latency(&pre, &p.chains);
        let (cute_ms, cute_nodes) = cute_latency(&p.chains);
        let same = p.chains.iter().all(cute_verified);
        println!(
            "{:>3}  {:<18} {:>6} | {:>10.3} {:>9.3} {:>9.3} {:>8} {:>7} | {:>10.5} {:>6} | {:>5}",
            x,
            p.name,
            p.chains.len(),
            a.total_ms,
            a.construct_ms,
            a.saturate_ms,
            a.nodes,
            a.rows,
            cute_ms,
            cute_nodes,
            same
        );
        csv.push_str(&format!(
            "{x},{},{},{:.4},{:.4},{:.4},{},{},{cute_ms:.6},{cute_nodes}\n",
            p.name,
            p.chains.len(),
            a.total_ms,
            a.construct_ms,
            a.saturate_ms,
            a.nodes,
            a.rows
        ));
    }

    let dir = "/private/tmp/claude-501/-Users-nolan-projects-anode/c09acde2-a3e9-4b5f-b340-489cfbaa9bf9/scratchpad";
    std::fs::create_dir_all(dir).unwrap();
    let out = format!("{dir}/sweep.csv");
    std::fs::write(&out, &csv).unwrap();
    println!("\nwrote {out}");
}
