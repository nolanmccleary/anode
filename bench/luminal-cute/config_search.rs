//! Tile config wired into Luminal's e-graph/genome search. The egglog match
//! rule mints one producer enode per tile config, so the matmul's output class
//! has several producers (one per config). With the competing fused-dot and
//! decomposed-reduce ops excluded, the ONLY producers are the config variants,
//! so Luminal's genome search must pick a config — and it profiles them and
//! keeps the fastest. This test checks a config is selected and the GEMM is
//! correct.
#![cfg(target_os = "macos")]
mod support;

use luminal::bufferize::BufferNode;
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::{CompileOptions, MetalRuntime, metal_registry_filtered};
use support::TestTransfers;

#[test]
fn luminal_search_selects_a_tile_config_and_is_correct() {
    let (m, n, k) = (256usize, 256usize, 256usize);
    let mut g = Graph::new();
    let a = g.tensor(vec![m, k], DType::F32);
    let b = g.tensor(vec![n, k], DType::F32);
    let out = a.matmul(b.t());

    let av: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
    let bv: Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();

    // Force the tiled-config matmul: exclude the simdgroup fused dot and the
    // decomposed reduction, so the ONLY producers of the matmul output are the
    // MatmulTiled config variants and the genome must choose among them.
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| {
            row.label() != "MulReduceSumGeneric" && row.label() != "ReduceSumGeneric"
        }),
    )
    .unwrap();

    // Enough exploration for the genome to sample several config variants.
    let opts = CompileOptions {
        generations: 6,
        generation_size: 8,
        trials: 3,
        seed: 0,
        search_log: false,
        ..CompileOptions::default()
    };
    rt.search(
        &Default::default(),
        &Default::default(),
        &[(a.id, av.clone().into()), (b.id, bv.clone().into())]
            .into_iter()
            .collect(),
        &opts,
    )
    .unwrap();

    // A MatmulTiled config variant was selected by the search.
    let used = rt
        .plan()
        .unwrap()
        .dag
        .node_weights()
        .any(|node| matches!(node, BufferNode::Compute { op, .. } if op.label() == "MatmulTiledGeneric"));
    assert!(used, "expected the genome search to select a MatmulTiled config");

    // Execute and verify vs a CPU reference.
    let mut arena = support::allocate_arena(&rt).unwrap();
    rt.upload(&mut arena, a.id, av.clone()).unwrap();
    rt.upload(&mut arena, b.id, bv.clone()).unwrap();
    rt.execute(arena.buffer()).unwrap();
    let got = rt.read_f32(&arena, out.id).unwrap();

    let mut want = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            for ki in 0..k {
                acc += av[mi * k + ki] as f64 * bv[ni * k + ki] as f64;
            }
            want[mi * n + ni] = acc as f32;
        }
    }
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(max_err < 1e-2, "GEMM wrong: max err {max_err}");
    eprintln!("Luminal genome search selected a MatmulTiled config; GEMM correct (err {max_err:.1e})");
}
