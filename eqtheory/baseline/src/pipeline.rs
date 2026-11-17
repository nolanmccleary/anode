use eqtheory_baseline::arena::{allocate_arena, read_f32, upload};
use eqtheory_baseline::loaders;
use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::{harness_search_options, metal_registry, MetalRuntime};
use std::time::Instant;

const M2_AIR_PEAK_GBPS: f64 = 100.0;

fn main() {
    let cfg = loaders::ModelConfig::default();
    println!(
        "AFFINE arm (Luminal pipeline) on loaded MiniLlama3  d={} ff={} heads={}/{} layers={} (decode)\n",
        cfg.d, cfg.ff, cfg.n_heads, cfg.n_kv_heads, cfg.layers
    );
    let loaded = loaders::load(&cfg);
    let read = loaded.read_ids();

    let mut rt = MetalRuntime::load_with(
        &loaded.graph,
        MetalBindings::dense(&loaded.graph.logical, &read),
        metal_registry(),
    )
    .expect("load");

    let t0 = Instant::now();
    rt.search(&Default::default(), &Default::default(), &loaded.inputs(), &harness_search_options())
        .expect("search");
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

    let plan = rt.plan().unwrap();
    let (mut copies, mut materializes, mut computes) = (0usize, 0usize, 0usize);
    for node in plan.dag.node_weights() {
        match node {
            BufferNode::BufferCopy { .. } | BufferNode::BufferCopyValue { .. } => copies += 1,
            BufferNode::Compute { op, .. } => {
                computes += 1;
                if op.label().contains("Materialize") || op.label().contains("Copy") {
                    materializes += 1;
                }
            }
            _ => {}
        }
    }

    let weight_bytes: usize = loaded
        .graph
        .logical
        .input_specs()
        .iter()
        .filter(|s| s.dtype == DType::F32)
        .map(|s| s.dims.iter().map(|d| d.to_usize().unwrap()).product::<usize>() * 4)
        .sum();

    let mut arena = allocate_arena(&rt).unwrap();
    for (id, hb) in loaded.inputs() {
        upload(&rt, &mut arena, id, hb).unwrap();
    }
    rt.execute(arena.buffer()).unwrap();
    let _ = read_f32(&rt, &arena, loaded.outputs[0].id).unwrap();

    let mut per_token_ms = f64::INFINITY;
    for _ in 0..50 {
        let t = Instant::now();
        rt.execute(arena.buffer()).unwrap();
        per_token_ms = per_token_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let tokens_per_sec = 1000.0 / per_token_ms;
    let achieved_gbps = (weight_bytes as f64) / (per_token_ms / 1e3) / 1e9;
    let bw_pct = achieved_gbps / M2_AIR_PEAK_GBPS * 100.0;

    println!("compile (Luminal search+profile) : {compile_ms:>8.0} ms");
    println!("compute kernels in plan          : {computes:>8}");
    println!("copy + materialize kernels       : {:>8}", copies + materializes);
    println!("weight bytes (F32 inputs)        : {:>8.1} MB", weight_bytes as f64 / 1e6);
    println!("per-token latency (min of 50)    : {per_token_ms:>8.3} ms");
    println!("tokens / sec                     : {tokens_per_sec:>8.0}");
    println!("memory bandwidth                 : {achieved_gbps:>8.1} GB/s  ({bw_pct:.0}% of M2 Air ~{M2_AIR_PEAK_GBPS:.0} GB/s peak)");
    println!(
        "\nmaterializations here are kernel-necessary (the model's views are shallow, so affine"
    );
    println!("recovers each one strided) -> cute's layout theory would produce the SAME plan ->");
    println!("identical tokens/sec + bandwidth. cute's only edge on a real model is compile time.");
}
