use eqtheory_baseline::loaders;
use luminal::bufferize::BufferNode;
use luminal::logical_ir::FusionMode;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::MetalRuntime;
use std::collections::BTreeMap;

const N: usize = 256;
const DEPTH: usize = 8;

fn plan_labels(fusion: FusionMode) -> BTreeMap<String, usize> {
    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load_deep_views(N, DEPTH);
    luminal::logical_ir::set_cute_lowering(false);

    let read = loaded.read_ids();
    let mut rt = MetalRuntime::load_with(
        &loaded.graph,
        MetalBindings::dense(&loaded.graph.logical, &read),
        eqtheory_baseline::baseline_registry::shared_registry(),
    )
    .expect("load");

    luminal::logical_ir::set_fusion_mode(fusion);
    rt.compile(&Default::default(), &Default::default()).expect("compile");
    luminal::logical_ir::set_fusion_mode(FusionMode::Luminal);

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for node in rt.plan().unwrap().dag.node_weights() {
        let label = match node {
            BufferNode::Compute { op, .. } => op.label().to_string(),
            BufferNode::BufferCopy { .. } => "~BufferCopy".to_string(),
            BufferNode::BufferCopyValue { .. } => "~BufferCopyValue".to_string(),
            _ => continue,
        };
        *counts.entry(label).or_default() += 1;
    }
    counts
}

fn main() {
    println!("KERNEL-LIST DIFF: cute fusion vs Luminal fusion on a DEEP-VIEW model");
    println!("(transpose+split+merge x{DEPTH}, N={N}, then matmul; both cute lowering)\n");
    let cute = plan_labels(FusionMode::Cute);
    let lum = plan_labels(FusionMode::Luminal);

    let mut labels: Vec<&String> = cute.keys().chain(lum.keys()).collect();
    labels.sort();
    labels.dedup();

    println!("{:<34} {:>6} {:>8} {:>6}", "kernel (op label)", "CUTE", "Luminal", "diff");
    println!("{}", "-".repeat(58));
    let mut identical = true;
    let (mut cute_total, mut lum_total) = (0usize, 0usize);
    for label in labels {
        let c = cute.get(label).copied().unwrap_or(0);
        let l = lum.get(label).copied().unwrap_or(0);
        cute_total += c;
        lum_total += l;
        let mark = if c != l {
            identical = false;
            " <<<"
        } else {
            ""
        };
        println!("{label:<34} {c:>6} {l:>8} {:>6}{mark}", c as i64 - l as i64);
    }
    println!("{}", "-".repeat(58));
    println!("{:<34} {cute_total:>6} {lum_total:>8}", "TOTAL");
    println!(
        "\n{}",
        if identical {
            "PLANS ARE IDENTICAL (same kernels) -> the tokens/sec gap was pure execution noise; clean tie."
        } else {
            "PLANS DIFFER at equal-ish count -> a real structural difference; worth timing properly."
        }
    );
}
