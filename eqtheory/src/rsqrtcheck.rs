use eqtheory::kernels::shared_registry;
use eqtheory::loaders;
use luminal::bufferize::BufferNode;
use luminal::logical_ir::FusionMode;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::{metal_registry, MetalRuntime, RegisteredOp};
use std::collections::BTreeMap;

fn counts(reg: Vec<RegisteredOp>) -> BTreeMap<String, usize> {
    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load_default();
    luminal::logical_ir::set_cute_lowering(false);
    let read = loaded.read_ids();
    let mut rt = MetalRuntime::load_with(
        &loaded.graph,
        MetalBindings::dense(&loaded.graph.logical, &read),
        reg,
    )
    .expect("load");
    luminal::logical_ir::set_fusion_mode(FusionMode::Luminal);
    rt.compile(&Default::default(), &Default::default())
        .expect("compile");
    luminal::logical_ir::set_fusion_mode(FusionMode::Luminal);
    let mut c = BTreeMap::new();
    let mut total = 0usize;
    for node in rt.plan().unwrap().dag.node_weights() {
        if let BufferNode::Compute { op, .. } = node {
            total += 1;
            let l = op.label().to_string();
            if l.contains("Sqrt") || l.contains("Recip") {
                *c.entry(l).or_default() += 1;
            }
        }
    }
    c.insert("TOTAL_compute".to_string(), total);
    c
}

fn main() {
    println!("RSQRT FUSION CHECK on loaded MiniLlama3 (default). Sqrt/Recip/Rsqrt label counts.\n");
    println!("baseline metal_registry (no rsqrt kernel):");
    for (k, v) in counts(metal_registry()) {
        println!("  {k:<34} {v}");
    }
    println!("\nshared_registry (with fused rsqrt kernel):");
    for (k, v) in counts(shared_registry()) {
        println!("  {k:<34} {v}");
    }
}
