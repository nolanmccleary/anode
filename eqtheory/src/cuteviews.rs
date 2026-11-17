use eqtheory::cute_egraph::{extract, saturate, Frame};
use eqtheory::loaders::{self, ModelConfig};
use eqtheory::cute_probe::probe;
use luminal::logical_ir::LogicalOp;
use luminal::prelude::petgraph::Direction;
use std::collections::{BTreeMap, BTreeSet};

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

fn eligible(
    g: &luminal::prelude::petgraph::stable_graph::StableDiGraph<
        luminal::logical_ir::LogicalNode,
        luminal::logical_ir::InputPort,
    >,
    id: luminal::prelude::NodeIndex,
) -> BTreeSet<&'static str> {
    fn feeds_reduce(
        g: &luminal::prelude::petgraph::stable_graph::StableDiGraph<
            luminal::logical_ir::LogicalNode,
            luminal::logical_ir::InputPort,
        >,
        id: luminal::prelude::NodeIndex,
        depth: usize,
    ) -> bool {
        g.neighbors_directed(id, Direction::Outgoing).any(|r| match g[r].op.constructor() {
            "LogicalReduceSum" => true,
            "LogicalIndexMapApply" if depth > 0 => feeds_reduce(g, r, depth - 1),
            _ => false,
        })
    }
    let mut out = BTreeSet::new();
    for c in g.neighbors_directed(id, Direction::Outgoing) {
        match g[c].op.constructor() {
            "LogicalMul" => {
                if feeds_reduce(g, c, 2) {
                    out.insert("tiled_matmul");
                }
                out.insert("elementwise");
            }
            "LogicalGather" | "LogicalScatter" => {
                out.insert("gather");
            }
            "LogicalIndexMapApply" => {}
            _ => {
                out.insert("elementwise");
            }
        }
    }
    out
}

fn main() {
    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load(&cfg());
    luminal::logical_ir::set_cute_lowering(false);

    let g = loaded.graph.logical.petgraph();
    let mut distinct: BTreeMap<(String, i64, String), usize> = BTreeMap::new();
    let (mut views, mut symbolic, mut unrepresentable) = (0usize, 0usize, 0usize);

    for id in g.node_indices() {
        let LogicalOp::IndexMapApply { entries } = &g[id].op else {
            continue;
        };
        views += 1;
        let src = g
            .neighbors_directed(id, Direction::Incoming)
            .next()
            .expect("view has a source");
        let out_dims: Option<Vec<i64>> = g[id].dims.iter().map(|d| Some(d.to_usize()? as i64)).collect();
        let src_dims: Option<Vec<i64>> = g[src].dims.iter().map(|d| Some(d.to_usize()? as i64)).collect();
        let (Some(out_dims), Some(src_dims)) = (out_dims, src_dims) else {
            symbolic += 1;
            continue;
        };
        assert_eq!(entries.len(), src_dims.len(), "one entry per source axis");
        let mut src_strides = vec![1i64; src_dims.len()];
        for k in (0..src_dims.len().saturating_sub(1)).rev() {
            src_strides[k] = src_strides[k + 1] * src_dims[k + 1];
        }
        let only_views = g
            .neighbors_directed(id, Direction::Outgoing)
            .all(|c| g[c].op.constructor() == "LogicalIndexMapApply");
        let elig: Vec<&str> = if only_views {
            vec!["interior"]
        } else {
            eligible(g, id).into_iter().collect()
        };
        match probe(entries, &out_dims, &src_strides) {
            Some(p) => *distinct.entry((p.layout, p.offset, elig.join("+"))).or_default() += 1,
            None => unrepresentable += 1,
        }
    }

    println!("REAL VIEW LAYOUTS: every IndexMapApply in the loaded MiniLlama3, probed + VERIFIED.");
    println!("each cute layout below denotes the identical address map as Luminal's MapEntry");
    println!("(exhaustive <=65536 cells, else 4096 samples).\n");
    println!(
        "{views} views: {} verified cute-representable, {symbolic} symbolic-extent skipped, {unrepresentable} not representable\n",
        distinct.values().sum::<usize>()
    );

    let frames = vec![
        Frame {
            name: "tiled_matmul",
            rank: 2,
            tiles: vec![vec![16, 16], vec![32, 32], vec![64, 64]],
        },
        Frame {
            name: "elementwise",
            rank: 1,
            tiles: vec![vec![1], vec![64], vec![256]],
        },
        Frame {
            name: "gather",
            rank: 1,
            tiles: vec![vec![1], vec![64], vec![256]],
        },
    ];

    println!(
        "{:<40} {:>3} {:>4} {:<26}   binding",
        "distinct view layout", "n", "rank", "consumers allow"
    );
    println!("{}", "-".repeat(110));
    for ((layout, offset, elig), count) in &distinct {
        let rank = layout.split(':').next().unwrap().matches(|c| c == ',').count() + 1;
        let tag = if *offset != 0 { format!("{layout} +{offset}") } else { layout.clone() };
        let allowed: Vec<Frame> = frames
            .iter()
            .filter(|f| elig.split('+').any(|e| e == f.name))
            .cloned()
            .collect();
        if elig == "interior" {
            println!("{tag:<40} {count:>3} {rank:>4} {elig:<26}   composes into downstream view (no separate read)");
            continue;
        }
        if allowed.is_empty() {
            println!("{tag:<40} {count:>3} {rank:>4} {elig:<26}   no eligible frame");
            continue;
        }
        let mut specs: Vec<String> = Vec::new();
        for ax in 0..rank {
            for t in [16, 32, 64, 256] {
                specs.push(format!("{ax}|{t}"));
            }
        }
        let spec_refs: Vec<&str> = specs.iter().map(|s| s.as_str()).collect();
        let bindings = saturate(layout, &spec_refs, 2);
        match extract(layout, &bindings, &allowed) {
            Some(c) => println!(
                "{tag:<40} {count:>3} {rank:>4} {elig:<26}   {:<13} {:?} waste {:.2}x",
                c.frame, c.tile, c.waste
            ),
            None => println!("{tag:<40} {count:>3} {rank:>4} {elig:<26}   eligible frame does not bind"),
        }
    }
}
