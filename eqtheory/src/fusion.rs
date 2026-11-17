use eqtheory::arena::{allocate_arena, read_f32, upload};
use eqtheory::loaders;
use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::logical_ir::FusionMode;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::MetalRuntime;
use std::time::Instant;

const M2_AIR_PEAK_GBPS: f64 = 100.0;

struct R {
    compute: usize,
    materializes: usize,
    search_ms: f64,
    per_token_ms: f64,
    tokens_per_sec: f64,
    bandwidth_gbps: f64,
    logits: Vec<f32>,
}

fn run(cute_lowering: bool, fusion: FusionMode) -> R {
    luminal::logical_ir::set_cute_lowering(cute_lowering);
    let loaded = loaders::load_default();
    luminal::logical_ir::set_cute_lowering(false);

    let read = loaded.read_ids();
    let mut rt = MetalRuntime::load_with(
        &loaded.graph,
        MetalBindings::dense(&loaded.graph.logical, &read),
        eqtheory::kernels::shared_registry(),
    )
    .expect("load");

    luminal::logical_ir::set_fusion_mode(fusion);
    let t0 = Instant::now();
    rt.compile(&Default::default(), &Default::default()).expect("compile");
    let search_ms = t0.elapsed().as_secs_f64() * 1e3;
    luminal::logical_ir::set_fusion_mode(FusionMode::Luminal);

    let weight_bytes: usize = loaded
        .graph
        .logical
        .input_specs()
        .iter()
        .filter(|s| s.dtype == DType::F32)
        .map(|s| s.dims.iter().map(|d| d.to_usize().unwrap()).product::<usize>() * 4)
        .sum();

    let plan = rt.plan().unwrap();
    let (mut compute, mut materializes) = (0usize, 0usize);
    for node in plan.dag.node_weights() {
        match node {
            BufferNode::BufferCopy { .. } | BufferNode::BufferCopyValue { .. } => materializes += 1,
            BufferNode::Compute { op, .. } => {
                compute += 1;
                if op.label().contains("Materialize") || op.label().contains("Copy") {
                    materializes += 1;
                }
            }
            _ => {}
        }
    }

    let mut arena = allocate_arena(&rt).unwrap();
    for (id, hb) in loaded.inputs() {
        upload(&rt, &mut arena, id, hb).unwrap();
    }
    rt.execute(arena.buffer()).unwrap();
    let logits = read_f32(&rt, &arena, loaded.outputs[0].id).unwrap();

    let mut per_token_ms = f64::INFINITY;
    for _ in 0..50 {
        let t = Instant::now();
        rt.execute(arena.buffer()).unwrap();
        per_token_ms = per_token_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let tokens_per_sec = 1000.0 / per_token_ms;
    let bandwidth_gbps = (weight_bytes as f64) / (per_token_ms / 1e3) / 1e9;

    R {
        compute,
        materializes,
        search_ms,
        per_token_ms,
        tokens_per_sec,
        bandwidth_gbps,
        logits,
    }
}

fn main() {
    println!("5-RUN MATRIX: lowering x fusion on loaded MiniLlama3 (same model, same kernels)\n");

    let runs = [
        ("1  cute-lower  + no fusion", true, FusionMode::None),
        ("2  affine-lower + no fusion", false, FusionMode::None),
        ("3  cute-lower  + CUTE fusion", true, FusionMode::Cute),
        ("4  affine-lower + Luminal fusion", false, FusionMode::Luminal),
        ("5  cute-lower  + Luminal fusion", true, FusionMode::Luminal),
    ];
    let results: Vec<R> = runs.iter().map(|(_, l, f)| run(*l, *f)).collect();

    let reference = &results[3].logits;
    let peak = reference.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));

    println!(
        "{:<34} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "run", "compute", "cpy+mat", "ms/tok", "tok/s", "GB/s", "match"
    );
    println!("{}", "-".repeat(84));
    let mut all_ok = true;
    for ((name, _, _), r) in runs.iter().zip(&results) {
        let max_abs = r
            .logits
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let ok = max_abs / peak < 1e-4;
        all_ok &= ok;
        println!(
            "{:<34} {:>7} {:>8} {:>8.3} {:>8.0} {:>8.1} {:>7}",
            name,
            r.compute,
            r.materializes,
            r.per_token_ms,
            r.tokens_per_sec,
            r.bandwidth_gbps,
            if ok { "match" } else { "DIFF" }
        );
    }
    let (cute, lum) = (&results[2], &results[3]);
    println!("\nall outputs identical to run #4 (affine+Luminal): {all_ok}");
    println!(
        "CUTE fusion vs Luminal fusion: {:.0} vs {:.0} tok/s  ({:.2}x),  {:.1} vs {:.1} GB/s  ({:.0}% vs {:.0}% of M2-Air peak)",
        cute.tokens_per_sec,
        lum.tokens_per_sec,
        cute.tokens_per_sec / lum.tokens_per_sec,
        cute.bandwidth_gbps,
        lum.bandwidth_gbps,
        cute.bandwidth_gbps / M2_AIR_PEAK_GBPS * 100.0,
        lum.bandwidth_gbps / M2_AIR_PEAK_GBPS * 100.0,
    );
}
