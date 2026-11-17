use eqtheory::loaders::{self, ModelConfig};
use luminal::prelude::petgraph::Direction;
use std::collections::BTreeMap;

fn cfg() -> ModelConfig {
    ModelConfig {
        vocab: 32_000,
        d: 1024,
        ff: 2752,
        n_heads: 16,
        n_kv_heads: 4,
        layers: 2,
        cache_capacity: 4,
    }
}

fn main() {
    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load(&cfg());
    luminal::logical_ir::set_cute_lowering(false);
    let g = loaded.graph.logical.petgraph();

    let covered: BTreeMap<&str, &str> = [
        ("LogicalIndexMapApply", "composes or materialize (copy)"),
        ("LogicalGather", "gather3"),
        ("LogicalAdd", "add2"),
        ("LogicalMul", "matmul when feeding ReduceSum, else missing ew-mul"),
    ]
    .into_iter()
    .collect();

    let (mut mm_mul, mut ew_mul, mut mm_rs, mut lone_rs) = (0usize, 0usize, 0usize, 0usize);
    for id in g.node_indices() {
        match g[id].op.constructor() {
            "LogicalMul" => {
                let feeds = g
                    .neighbors_directed(id, Direction::Outgoing)
                    .any(|r| g[r].op.constructor() == "LogicalReduceSum");
                if feeds { mm_mul += 1 } else { ew_mul += 1 }
            }
            "LogicalReduceSum" => {
                let fed = g
                    .neighbors_directed(id, Direction::Incoming)
                    .any(|r| g[r].op.constructor() == "LogicalMul");
                if fed { mm_rs += 1 } else { lone_rs += 1 }
            }
            _ => {}
        }
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut dtypes: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for id in g.node_indices() {
        let c = g[id].op.constructor().to_string();
        *counts.entry(c.clone()).or_default() += 1;
        *dtypes
            .entry(c)
            .or_default()
            .entry(format!("{:?}", g[id].dtype))
            .or_default() += 1;
    }

    println!("OP CENSUS: every logical op in the loaded MiniLlama3 (2 layers, d=1024).\n");
    println!("{:<28} {:>5}  {:<28} {}", "constructor", "n", "dtypes", "shared-kernel status");
    println!("{}", "-".repeat(100));
    let (mut have, mut miss) = (0usize, 0usize);
    for (c, n) in &counts {
        let dt: Vec<String> = dtypes[c].iter().map(|(k, v)| format!("{k}x{v}")).collect();
        let status = covered.get(c.as_str()).copied().unwrap_or("MISSING");
        if status == "MISSING" {
            miss += n;
        } else {
            have += n;
        }
        println!("{c:<28} {n:>5}  {:<28} {status}", dt.join(" "));
    }
    println!("{}", "-".repeat(100));
    println!("{have} nodes with a shared-kernel story, {miss} nodes MISSING coverage\n");
    println!("Mul split: {mm_mul} matmul-pattern (pair with ReduceSum -> tiled_matmul), {ew_mul} elementwise");
    println!("ReduceSum split: {mm_rs} matmul-paired (absorbed), {lone_rs} standalone (need reduce kernel)");
}
