use eqtheory::executor::Exec;
use eqtheory::loaders::{self, ModelConfig};
use std::io::Write;
use std::path::Path;

fn cfg() -> ModelConfig {
    ModelConfig {
        vocab: 32_000,
        d: 1024,
        ff: 2752,
        n_heads: 16,
        n_kv_heads: 4,
        layers: 8,
        cache_capacity: 4,
    }
}

fn main() {
    let out_dir = std::env::args().nth(1).expect("usage: export_model <dir>");
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out).unwrap();

    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load(&cfg());
    luminal::logical_ir::set_cute_lowering(false);
    let inputs = loaded.inputs();

    let mut manifest = std::fs::File::create(out.join("manifest.txt")).unwrap();
    for (i, spec) in loaded.graph.logical.input_specs().iter().enumerate() {
        let hb = &inputs[&spec.id];
        let dims: Vec<String> = spec
            .dims
            .iter()
            .map(|d| d.to_usize().unwrap().to_string())
            .collect();
        let fname = format!("in{i}.bin");
        let bytes: Vec<u8> = match spec.dtype {
            luminal::dtype::DType::F32 => hb
                .as_f32()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            luminal::dtype::DType::Int => hb
                .as_i32()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            other => panic!("dtype {other:?}"),
        };
        std::fs::write(out.join(&fname), bytes).unwrap();
        writeln!(
            manifest,
            "input|{}|{:?}|{}|{fname}",
            spec.label,
            spec.dtype,
            dims.join(",")
        )
        .unwrap();
    }

    let mut ex = Exec::new(loaded.graph.logical.petgraph());
    ex.run(&inputs);
    ex.replay();
    for (k, o) in loaded.outputs.iter().enumerate() {
        let vals = ex.read_f32(o.id);
        let fname = format!("out{k}.bin");
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join(&fname), bytes).unwrap();
        writeln!(manifest, "output|{k}|F32|{}|{fname}", vals.len()).unwrap();
    }
    println!("exported to {out_dir}");
}
