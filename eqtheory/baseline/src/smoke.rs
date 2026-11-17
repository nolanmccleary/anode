use eqtheory_baseline::arena::{allocate_arena, read_f32, upload};
use eqtheory_baseline::loaders;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::{harness_search_options, metal_registry, MetalRuntime};
use std::time::Instant;

fn main() {
    let cfg = loaders::ModelConfig::default();
    println!(
        "loading real model: MiniLlama3  vocab={} d={} ff={} heads={} kv_heads={} layers={} (decode, M2-Air default)",
        cfg.vocab, cfg.d, cfg.ff, cfg.n_heads, cfg.n_kv_heads, cfg.layers
    );
    let loaded = loaders::load(&cfg);
    let read = loaded.read_ids();

    let mut rt = MetalRuntime::load_with(
        &loaded.graph,
        MetalBindings::dense(&loaded.graph.logical, &read),
        metal_registry(),
    )
    .expect("load real model");

    let t0 = Instant::now();
    rt.search(
        &Default::default(),
        &Default::default(),
        &loaded.inputs(),
        &harness_search_options(),
    )
    .expect("search");
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut arena = allocate_arena(&rt).unwrap();
    for (id, hb) in loaded.inputs() {
        upload(&rt, &mut arena, id, hb).unwrap();
    }
    rt.execute(arena.buffer()).unwrap();
    let logits = read_f32(&rt, &arena, loaded.outputs[0].id).unwrap();

    println!("compiled + executed on Metal in {compile_ms:.0} ms (search+stage)");
    println!(
        "logits: len {}  sample {:?}",
        logits.len(),
        &logits[..logits.len().min(5)]
    );
    println!("loader works: real model loaded, searched, executed, read back.");
}
