//! Control experiment: are the search pathologies (budget-sensitive plans with a
//! redundant transpose copy; not converging to the fastest option) intrinsic to
//! Luminal's stock search, or introduced by my new config-variant ops?
//!
//! Same graph (a @ b.t()), same genome search, two op sets:
//!   STOCK: exclude my ops -> matmul is the stock MulReduceSum (simdgroup tiled).
//!   MINE:  exclude the stock reduce -> matmul is a MatmulTiled config variant.
//! Swept over {low, high} search budget and several seeds. We report delivered
//! GFLOP/s (from best_nanos), the matmul op chosen, and whether the plan carried
//! a redundant layout-copy/materialize kernel.
//!
//! Run: cargo +1.99.1 run --release --example baseline_control -p luminal_metal
#![cfg(target_os = "macos")]

use luminal::bufferize::BufferNode;
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::{CompileOptions, MetalRuntime, metal_registry_filtered};
use rustc_hash::FxHashMap;

fn run(m: usize, n: usize, k: usize, mine: bool, gens: usize, size: usize, trials: usize, seed: u64)
    -> Option<(f64, String, bool)> {
    let mut g = Graph::new();
    let a = g.tensor(vec![m, k], DType::F32);
    let b = g.tensor(vec![n, k], DType::F32);
    let _out = a.matmul(b.t());
    let reg = if mine {
        // only the config-variant matmul can produce the output
        metal_registry_filtered(|r| r.label() != "MulReduceSumGeneric" && r.label() != "ReduceSumGeneric")
    } else {
        // pure stock: remove my ops entirely
        metal_registry_filtered(|r| r.label() != "MatmulTiledGeneric" && r.label() != "MatmulBiasGeneric")
    };
    let mut rt = MetalRuntime::load_with_registry(&g, reg).ok()?;
    let av: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
    let bv: Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
    let inputs: FxHashMap<_, _> = [(a.id, av.into()), (b.id, bv.into())].into_iter().collect();
    let opts = CompileOptions {
        generations: gens, generation_size: size, trials, seed, search_log: false,
        ..CompileOptions::default()
    };
    let outcome = rt.search(&Default::default(), &Default::default(), &inputs, &opts).ok()?;
    let plan = rt.plan().unwrap();
    let ops: Vec<String> = plan.dag.node_weights().filter_map(|nd| match nd {
        BufferNode::Compute { op, .. } => Some(op.label().to_string()),
        _ => None,
    }).collect();
    let matmul = ops.iter().find(|l| l.contains("MatmulTiled") || l.contains("MulReduceSum"))
        .cloned().unwrap_or_else(|| "?".into());
    let has_copy = ops.iter().any(|l| l.contains("Copy") || l.contains("Materialize"));
    let gflops = 2.0 * m as f64 * n as f64 * k as f64 / (outcome.best_nanos as f64);
    Some((gflops, matmul, has_copy))
}

fn main() {
    let (m, n, k) = (1024usize, 1024, 1024);
    println!("a @ b.t()  {m}x{n}x{k}  (delivered GFLOP/s from best_nanos)\n");
    for (blabel, gens, size, trials) in [("LOW  budget", 8usize, 10usize, 4usize),
                                         ("HIGH budget", 20, 16, 12)] {
        println!("== {blabel} (gens={gens} size={size} trials={trials}) ==");
        for mine in [false, true] {
            let arm = if mine { "MINE (MatmulTiled configs)" } else { "STOCK (MulReduceSum)      " };
            print!("  {arm}: ");
            let mut best = 0f64;
            let mut summary = Vec::new();
            for seed in 0..4u64 {
                if let Some((g, op, copy)) = run(m, n, k, mine, gens, size, trials, seed) {
                    best = best.max(g);
                    summary.push(format!("s{seed}:{g:.0}{}{}", if copy { "+copy" } else { "" },
                        if op.contains("Tiled") { "" } else { "*" }));
                }
            }
            println!("best {best:5.0} | per-seed [{}]", summary.join(" "));
        }
        println!();
    }
    println!("+copy = plan inserted a redundant transpose materialize kernel.");
    println!("* = matmul ran as stock MulReduceSum (simdgroup).  best = max over seeds.");
}
