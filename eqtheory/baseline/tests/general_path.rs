use eqtheory_baseline::executor::Exec;
use eqtheory_baseline::loaders;

#[test]
fn general_path_exists_and_matches() {
    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load_default();
    luminal::logical_ir::set_cute_lowering(false);
    let inputs = loaded.inputs();

    unsafe { std::env::set_var("GENERAL_ONLY", "1") };
    let mut general = Exec::new(loaded.graph.logical.petgraph());
    general.run(&inputs);
    general.replay();
    let g_out: Vec<Vec<f32>> = loaded.outputs.iter().map(|o| general.read_f32(o.id)).collect();
    assert_eq!(general.fused_chains.get(), 0, "general mode must not fuse");
    let weight_bytes: usize = inputs
        .values()
        .map(|h| h.as_f32().map(|v| v.len() * 4).unwrap_or(0))
        .sum();
    assert!(
        general.alloc_total.get() < weight_bytes,
        "general path intermediates ({}) must stay below weight bytes ({}) — a sole-path fused kernel is hiding a blowup",
        general.alloc_total.get(),
        weight_bytes
    );
    unsafe { std::env::remove_var("GENERAL_ONLY") };

    let mut fused = Exec::new(loaded.graph.logical.petgraph());
    fused.run(&inputs);
    fused.replay();
    for (k, o) in loaded.outputs.iter().enumerate() {
        let f = fused.read_f32(o.id);
        let peak = f.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));
        let d = g_out[k]
            .iter()
            .zip(&f)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            d / peak < 1e-3,
            "output {k}: general vs fused rel diff {} — fused kernels drifted from the primitive semantics",
            d / peak
        );
    }
}
