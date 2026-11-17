use eqtheory_baseline::affine_arm::AffineExec;
use eqtheory_baseline::arena::{allocate_arena, read_f32, upload};
use eqtheory_baseline::executor::Exec;
use eqtheory_baseline::loaders::{self, ModelConfig};
use luminal_metal::bindings::MetalBindings;
use luminal_metal::MetalRuntime;
use std::time::Instant;

struct Row {
    arm: &'static str,
    compile_ms: f64,
    tok_s: f64,
    peak_mb: f64,
    traffic_mb: f64,
    dispatches: usize,
    verdict: String,
}

fn time_replays(mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn verdict(ours: &[Vec<f32>], oracle: &[Vec<f32>]) -> String {
    let mut worst = 0f64;
    for (a, b) in ours.iter().zip(oracle) {
        let peak = b.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        worst = worst.max((d / peak.max(1e-30)) as f64);
    }
    if worst < 1e-4 {
        format!("MATCH rel {worst:.1e}")
    } else {
        format!("MISMATCH rel {worst:.1e}")
    }
}

fn run_config(name: &str, cfg: Option<ModelConfig>) {
    println!("===== {name} =====");

    luminal::logical_ir::set_cute_lowering(true);
    let loaded_cute = match cfg {
        Some(c) => loaders::load(&c),
        None => loaders::load_default(),
    };
    luminal::logical_ir::set_cute_lowering(false);
    let inputs_cute = loaded_cute.inputs();

    let t0 = Instant::now();
    let mut ex = Exec::new(loaded_cute.graph.logical.petgraph());
    ex.run(&inputs_cute);
    let cute_compile = t0.elapsed().as_secs_f64() * 1e3;
    let cute_exec = time_replays(|| ex.replay());
    let cute_out: Vec<Vec<f32>> = loaded_cute.outputs.iter().map(|o| ex.read_f32(o.id)).collect();
    let cute_row_partial = (
        cute_compile,
        1000.0 / cute_exec,
        ex.alloc_total.get() as f64 / 1e6,
        ex.traffic_bytes() as f64 / 1e6,
        ex.dispatches.get(),
    );
    drop(ex);

    luminal::logical_ir::set_cute_lowering(false);
    let loaded = match cfg {
        Some(c) => loaders::load(&c),
        None => loaders::load_default(),
    };
    let inputs = loaded.inputs();
    let read_ids = loaded.read_ids();
    let specs = loaded.graph.logical.input_specs();

    let mut rt = MetalRuntime::load_with(
        &loaded.graph,
        MetalBindings::dense(&loaded.graph.logical, &read_ids),
        eqtheory_baseline::baseline_registry::shared_registry(),
    )
    .expect("load");
    let t1 = Instant::now();
    rt.compile(&Default::default(), &Default::default()).expect("compile");
    let affine_compile = t1.elapsed().as_secs_f64() * 1e3;
    let plan = rt.plan().unwrap().clone();
    let t2 = Instant::now();
    let mut ax = AffineExec::new();
    ax.build(&plan, &specs, &inputs);
    let affine_map = t2.elapsed().as_secs_f64() * 1e3;
    let affine_exec = time_replays(|| ax.replay());
    let affine_out: Vec<Vec<f32>> =
        (0..loaded.outputs.len()).map(|k| ax.read_output(k)).collect();

    let mut arena = allocate_arena(&rt).unwrap();
    for (id, hb) in inputs.iter() {
        upload(&rt, &mut arena, *id, hb.clone()).unwrap();
    }
    rt.execute(arena.buffer()).unwrap();
    let oracle: Vec<Vec<f32>> = loaded
        .outputs
        .iter()
        .map(|o| read_f32(&rt, &arena, o.id).unwrap())
        .collect();

    let rows = [
        Row {
            arm: "cute",
            compile_ms: cute_row_partial.0,
            tok_s: cute_row_partial.1,
            peak_mb: cute_row_partial.2,
            traffic_mb: cute_row_partial.3,
            dispatches: cute_row_partial.4,
            verdict: verdict(&cute_out, &oracle),
        },
        Row {
            arm: "affine",
            compile_ms: affine_compile + affine_map,
            tok_s: 1000.0 / affine_exec,
            peak_mb: ax.peak_bytes as f64 / 1e6,
            traffic_mb: ax.traffic_bytes() as f64 / 1e6,
            dispatches: ax.dispatches,
            verdict: verdict(&affine_out, &oracle),
        },
    ];

    println!(
        "{:<8} {:>12} {:>8} {:>10} {:>12} {:>10}   vs oracle",
        "arm", "compile(ms)", "tok/s", "peak(MB)", "MB/token", "dispatches"
    );
    println!("{}", "-".repeat(84));
    for r in rows {
        println!(
            "{:<8} {:>12.1} {:>8.0} {:>10.1} {:>12.1} {:>10}   {}",
            r.arm, r.compile_ms, r.tok_s, r.peak_mb, r.traffic_mb, r.dispatches, r.verdict
        );
    }
    println!(
        "note: affine compile = luminal saturation+extraction {affine_compile:.1}ms + plan->shared-kernel map {affine_map:.1}ms"
    );
    println!("note: peak = cute total allocated (no reuse) vs affine planned alloc/free peak; MB/token = logical kernel read+write traffic\n");
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
