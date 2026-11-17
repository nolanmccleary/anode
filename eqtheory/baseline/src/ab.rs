use eqtheory_baseline::arena::{allocate_arena, read_f32, upload};
use eqtheory_baseline::loaders;
use luminal::bufferize::BufferNode;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::{harness_search_options, metal_registry, MetalRuntime};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
enum Ruleset {
    Affine,
    Cute,
}

struct Arm {
    search_ms: f64,
    kernel_compilations: usize,
    materializes: usize,
    per_token_ms: f64,
    tokens_per_sec: f64,
    logits: Vec<f32>,
}

fn run_arm(ruleset: Ruleset) -> Arm {
    luminal::logical_ir::set_cute_lowering(ruleset == Ruleset::Cute);
    let loaded = loaders::load_default();
    luminal::logical_ir::set_cute_lowering(false);

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
    let search_ms = t0.elapsed().as_secs_f64() * 1e3;

    let kernel_compilations = rt.graph_stats().map(|s| s.kernel_compilations as usize).unwrap_or(0);

    let plan = rt.plan().unwrap();
    let mut materializes = 0usize;
    for node in plan.dag.node_weights() {
        match node {
            BufferNode::BufferCopy { .. } | BufferNode::BufferCopyValue { .. } => materializes += 1,
            BufferNode::Compute { op, .. } => {
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

    Arm {
        search_ms,
        kernel_compilations,
        materializes,
        per_token_ms,
        tokens_per_sec: 1000.0 / per_token_ms,
        logits,
    }
}

fn main() {
    println!("REAL pipeline-vs-pipeline: cute vs affine LAYOUT THEORY on a loaded model");
    println!("same model (MiniLlama3), same Luminal kernels, switch = feature cute (gated).\n");

    let affine = run_arm(Ruleset::Affine);
    let cute = run_arm(Ruleset::Cute);

    let max_abs = affine
        .logits
        .iter()
        .zip(&cute.logits)
        .map(|(a, c)| (a - c).abs())
        .fold(0f32, f32::max);
    let peak = affine.logits.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));

    println!("{:<26} {:>14} {:>14}", "metric", "AFFINE", "CUTE");
    println!("{}", "-".repeat(56));
    println!("{:<26} {:>14.0} {:>14.0}", "search / compile (ms)", affine.search_ms, cute.search_ms);
    println!("{:<26} {:>14} {:>14}", "kernel compilations", affine.kernel_compilations, cute.kernel_compilations);
    println!("{:<26} {:>14} {:>14}", "copy+materialize kernels", affine.materializes, cute.materializes);
    println!("{:<26} {:>14.3} {:>14.3}", "per-token latency (ms)", affine.per_token_ms, cute.per_token_ms);
    println!("{:<26} {:>14.0} {:>14.0}", "tokens / sec", affine.tokens_per_sec, cute.tokens_per_sec);
    println!("\noutput agreement (cute vs affine): max abs {max_abs:.3e}  ({:.2e} relative)", max_abs / peak);
    println!(
        "{}",
        if max_abs / peak < 1e-4 { "OUTPUTS MATCH — cute lowering is semantically identical" } else { "MISMATCH — cute canonicalizer bug" }
    );
}
