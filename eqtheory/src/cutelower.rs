use eqtheory::cute_egraph::{extract, saturate, Frame};
use eqtheory::loaders::{self, ModelConfig};
use luminal::dtype::DType;
use std::collections::BTreeMap;

fn cfg() -> ModelConfig {
    ModelConfig {
        vocab: 32_000,
        d: 1024,
        ff: 2752,
        n_heads: 16,
        n_kv_heads: 4,
        layers: 4,
        cache_capacity: 4,
    }
}

fn main() {
    let specs = ["0|16", "1|16", "0|32", "1|32", "0|64", "1|64", "0|128", "1|128"];
    let frames = vec![Frame {
        name: "tiled_matmul",
        rank: 2,
        tiles: vec![vec![16, 16], vec![32, 32], vec![64, 64], vec![128, 128]],
    }];

    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load(&cfg());
    luminal::logical_ir::set_cute_lowering(false);

    let mut shapes: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for s in loaded.graph.logical.input_specs() {
        if s.dtype == DType::F32 && s.dims.len() == 2 {
            let m = s.dims[0].to_usize().unwrap();
            let k = s.dims[1].to_usize().unwrap();
            *shapes.entry((m, k)).or_default() += 1;
        }
    }

    println!("CUTE LOWERING: the selector chooses kernels by layout eq-class on the loaded model.");
    println!("operand layouts are the model's real rank-2 F32 tensors; no Luminal matchers run.\n");
    println!(
        "{:<16} {:>6} {:>10} {:>8}   {}",
        "operand", "count", "tile", "waste", "access"
    );
    println!("{}", "-".repeat(76));

    let (mut total, mut serviced, mut waste_sum) = (0usize, 0usize, 0.0f64);
    for ((m, k), count) in &shapes {
        total += count;
        let data = format!("({m},{k}):({k},1)");
        let bindings = saturate(&data, &specs, 2);
        match extract(&data, &bindings, &frames) {
            Some(c) => {
                serviced += count;
                waste_sum += c.waste * *count as f64;
                println!(
                    "({m},{k})         {count:>6} {:>10} {:>8.3}   {}",
                    format!("{}x{}", c.tile[0], c.tile[1]),
                    c.waste,
                    c.access
                );
            }
            None => println!("({m},{k})         {count:>6} {:>10}", "none"),
        }
    }

    println!("{}", "-".repeat(76));
    println!(
        "{serviced}/{total} operand tensors serviced by one tiled_matmul via composition; mean waste {:.3}x",
        waste_sum / serviced.max(1) as f64
    );
}
