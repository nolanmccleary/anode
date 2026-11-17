use eqtheory_baseline::loaders;
use luminal::bufferize::BufferNode;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::MetalRuntime;
use std::collections::BTreeMap;

fn main() {
    luminal::logical_ir::set_cute_lowering(false);
    let loaded = loaders::load_default();
    let read = loaded.read_ids();
    let mut rt = MetalRuntime::load_with(
        &loaded.graph,
        MetalBindings::dense(&loaded.graph.logical, &read),
        eqtheory_baseline::baseline_registry::shared_registry(),
    )
    .expect("load");
    rt.compile(&Default::default(), &Default::default()).expect("compile");
    let plan = rt.plan().unwrap();

    let mut labels: BTreeMap<String, usize> = BTreeMap::new();
    let (mut copies, mut copy_values, mut no_extents) = (0usize, 0usize, 0usize);
    for node in plan.dag.node_weights() {
        match node {
            BufferNode::Compute { op, operand_info, result_info, .. } => {
                *labels.entry(op.label().to_string()).or_default() += 1;
                for s in operand_info.iter().chain(result_info) {
                    if s.layout.literal_extents().is_none() {
                        no_extents += 1;
                    }
                }
            }
            BufferNode::BufferCopy { .. } => copies += 1,
            BufferNode::BufferCopyValue { .. } => copy_values += 1,
            _ => {}
        }
    }
    println!("AFFINE PLAN CENSUS (Luminal pipeline, shared_registry, default model)\n");
    for (l, n) in &labels {
        println!("  {l:<34} {n}");
    }
    println!("\n  BufferCopy {copies}, BufferCopyValue {copy_values}, slots missing literal extents: {no_extents}");

    let mut nbuf = 0usize;
    let mut no_size = 0usize;
    for (_, b) in plan.buffers.iter() {
        nbuf += 1;
        let d = format!("{b:?}");
        if !d.contains("size") && !d.contains("bytes") && !d.contains("span") {
            no_size += 1;
        }
    }
    println!("  buffers: {nbuf} (debug lacking size-ish field: {no_size})");
    if let Some(b) = plan.buffers.values().next() {
        println!("  sample buffer debug: {b:?}");
    }
    for node in plan.dag.node_weights() {
        if let BufferNode::BufferOutput { slots } = node {
            println!("  output slots: {}", slots.len());
            if let Some(s) = slots.first() {
                println!("  sample output binding: {s:?}");
            }
            break;
        }
    }
}
