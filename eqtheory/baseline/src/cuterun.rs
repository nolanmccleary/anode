use eqtheory_baseline::arena::{allocate_arena, read_f32, upload};
use eqtheory_baseline::executor::Exec;
use eqtheory_baseline::loaders::{self, ModelConfig};
use luminal_metal::bindings::MetalBindings;
use luminal_metal::MetalRuntime;
use std::time::Instant;

fn run_config(name: &str, cfg: Option<ModelConfig>) {
    println!("== {name} ==");
    luminal::logical_ir::set_cute_lowering(true);
    let loaded = match cfg {
        Some(c) => loaders::load(&c),
        None => loaders::load_default(),
    };
    luminal::logical_ir::set_cute_lowering(false);
    let inputs = loaded.inputs();
    let read_ids = loaded.read_ids();

    let t0 = Instant::now();
    let mut ex = Exec::new(loaded.graph.logical.petgraph());
    ex.run(&inputs);
    let plan_ms = t0.elapsed().as_secs_f64() * 1e3;
    ex.replay();
    let mut exec_ms = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        ex.replay();
        exec_ms = exec_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let cute_ms = plan_ms;
    let cute_logits = ex.read_f32(loaded.outputs[0].id);
    println!("plan {plan_ms:.1} ms, execute {exec_ms:.3} ms/token -> {:.0} tok/s", 1000.0/exec_ms);
    println!(
        "cute executor: {} dispatches ({} tiled matmuls), {} materializes, {} inline strided reads, {} fused chains, {} unclaimed ew edges, {} static idx folds, {:.1} ms wall",
        ex.dispatches.get(), ex.matmuls_paired.get(), ex.materializes.get(), ex.inline_reads.get(), ex.fused_chains.get(), ex.generic_ew_edges.get(), ex.static_idx_folds.get(), cute_ms
    );

    let mut rt = MetalRuntime::load_with(
        &loaded.graph,
        MetalBindings::dense(&loaded.graph.logical, &read_ids),
        eqtheory_baseline::baseline_registry::shared_registry(),
    )
    .expect("oracle load");
    rt.compile(&Default::default(), &Default::default()).expect("oracle compile");
    let mut arena = allocate_arena(&rt).unwrap();
    for (id, hb) in inputs.iter() {
        upload(&rt, &mut arena, *id, hb.clone()).unwrap();
    }
    rt.execute(arena.buffer()).unwrap();
    let oracle_logits = read_f32(&rt, &arena, loaded.outputs[0].id).unwrap();

    assert_eq!(cute_logits.len(), oracle_logits.len(), "logit lengths");
    let peak = oracle_logits.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));
    let max_abs = cute_logits
        .iter()
        .zip(&oracle_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "logits vs Luminal oracle: max abs diff {max_abs:.3e}, peak {peak:.3e}, rel {:.3e} -> {}",
        max_abs / peak,
        if max_abs / peak < 1e-4 { "MATCH" } else { "MISMATCH" }
    );

    if std::env::var("VALS").is_ok() {
        let c1 = ex.read_f32(loaded.outputs[1].id);
        println!("cute k0 row [128..132] = {:?}", &c1[128..132]);
    }
    for (k, out) in loaded.outputs.iter().enumerate().skip(1) {
        let c = ex.read_f32(out.id);
        let o = read_f32(&rt, &arena, out.id).unwrap();
        let d = c.iter().zip(&o).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let p = o.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));
        let ok = if p > 0.0 { d / p < 1e-4 } else { d == 0.0 };
        println!("cache out {k}: max abs diff {d:.3e} -> {}", if ok { "MATCH" } else { "MISMATCH" });
    }
    println!();
}

fn main() {
    run_config("default MiniLlama3 (d=512, 2 layers)", None);
    run_config(
        "fat MiniLlama3 (d=1024, ff=2752, heads 16/4, 8 layers)",
        Some(ModelConfig {
            vocab: 32_000,
            d: 1024,
            ff: 2752,
            n_heads: 16,
            n_kv_heads: 4,
            layers: 8,
            cache_capacity: 4,
        }),
    );
}
