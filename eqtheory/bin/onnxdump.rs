fn main() {
    let path = std::env::args().nth(1).expect("usage: onnxdump <model.onnx>");
    let bytes = std::fs::read(&path).unwrap();
    let m = <eqtheory::onnx_proto::ModelProto as prost::Message>::decode(&bytes[..]).unwrap();
    let g = m.graph.unwrap();
    use std::collections::BTreeMap;
    let mut ops: BTreeMap<String, usize> = BTreeMap::new();
    for n in &g.node {
        *ops.entry(n.op_type.clone().unwrap_or_default()).or_default() += 1;
    }
    println!("nodes {} inits {}", g.node.len(), g.initializer.len());
    for (k, v) in &ops {
        println!("  {k} x{v}");
    }
}
