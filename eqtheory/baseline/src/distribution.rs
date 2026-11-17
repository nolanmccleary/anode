use eqtheory_baseline::arena::{allocate_arena, read_f32, upload};
use eqtheory_baseline::loaders::{self, ModelConfig};
use luminal::bufferize::BufferNode;
use luminal::graph::Graph;
use luminal::logical_ir::FusionMode;
use luminal::prelude::NodeIndex;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::{harness_search_options, MetalRuntime};
use std::time::Instant;

const N: usize = 12;

fn fat() -> ModelConfig {
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

struct Sample {
    copymat: usize,
    compute: usize,
    tokens_per_sec: f64,
}

fn search_once(
    graph: &Graph,
    read: &[NodeIndex],
    inputs: &luminal::prelude::FxHashMap<NodeIndex, luminal_metal::HostBuffer>,
    fusion: FusionMode,
    out: NodeIndex,
) -> Sample {
    let mut rt = MetalRuntime::load_with(
        graph,
        MetalBindings::dense(&graph.logical, read),
        eqtheory_baseline::baseline_registry::shared_registry(),
    )
    .expect("load");
    luminal::logical_ir::set_fusion_mode(fusion);
    rt.search(&Default::default(), &Default::default(), inputs, &harness_search_options())
        .expect("search");
    luminal::logical_ir::set_fusion_mode(FusionMode::Luminal);

    let (mut compute, mut copymat) = (0usize, 0usize);
    for node in rt.plan().unwrap().dag.node_weights() {
        match node {
            BufferNode::BufferCopy { .. } | BufferNode::BufferCopyValue { .. } => copymat += 1,
            BufferNode::Compute { op, .. } => {
                compute += 1;
                if op.label().contains("Materialize") || op.label().contains("Copy") {
                    copymat += 1;
                }
            }
            _ => {}
        }
    }

    let mut arena = allocate_arena(&rt).unwrap();
    for (id, hb) in inputs.iter() {
        upload(&rt, &mut arena, *id, hb.clone()).unwrap();
    }
    rt.execute(arena.buffer()).unwrap();
    let _ = read_f32(&rt, &arena, out);
    let mut per_token_ms = f64::INFINITY;
    for _ in 0..30 {
        let t = Instant::now();
        rt.execute(arena.buffer()).unwrap();
        per_token_ms = per_token_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    Sample { copymat, compute, tokens_per_sec: 1000.0 / per_token_ms }
}

fn stats(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = v[0];
    let med = v[v.len() / 2];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (min, med, mean)
}

fn collect(fusion: FusionMode) -> Vec<Sample> {
    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load(&fat());
    luminal::logical_ir::set_cute_lowering(false);
    let read = loaded.read_ids();
    let inputs = loaded.inputs();
    let out = loaded.outputs[0].id;
    (0..N)
        .map(|_| search_once(&loaded.graph, &read, &inputs, fusion, out))
        .collect()
}

fn report(name: &str, s: &[Sample]) {
    let mut cm: Vec<f64> = s.iter().map(|x| x.copymat as f64).collect();
    let mut cp: Vec<f64> = s.iter().map(|x| x.compute as f64).collect();
    let mut tk: Vec<f64> = s.iter().map(|x| x.tokens_per_sec).collect();
    let (cm_min, cm_med, cm_mean) = stats(&mut cm);
    let (cp_min, cp_med, _cp_mean) = stats(&mut cp);
    let (_tk_min, tk_med, tk_mean) = stats(&mut tk);
    println!(
        "{name:<18} copy+mat[min/med/mean {cm_min:.0}/{cm_med:.0}/{cm_mean:.1}]  compute[min/med {cp_min:.0}/{cp_med:.0}]  tok/s[med/mean {tk_med:.0}/{tk_mean:.0}]"
    );
}

fn main() {
    let f = fat();
    println!(
        "DISTRIBUTION over {N} fusion-aware searches/config. fat MiniLlama3 d={} ff={} heads={}/{} layers={}\n",
        f.d, f.ff, f.n_heads, f.n_kv_heads, f.layers
    );
    let cute = collect(FusionMode::Cute);
    let luminal = collect(FusionMode::Luminal);
    report("CUTE fusion", &cute);
    report("Luminal fusion", &luminal);

    let cute_cm: f64 = cute.iter().map(|x| x.copymat as f64).sum::<f64>() / N as f64;
    let lum_cm: f64 = luminal.iter().map(|x| x.copymat as f64).sum::<f64>() / N as f64;
    println!(
        "\nmean copy+mat: cute {cute_cm:.1} vs Luminal {lum_cm:.1}  (delta {:+.1})",
        cute_cm - lum_cm
    );
}
