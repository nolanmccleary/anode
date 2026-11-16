use eqtheory::views::{
    affine_saturate, affine_seed, backend_copies, build_viewed_matmul, cute_lower, cute_verified,
    deep_round_trip, preamble, Model, Op,
};
use luminal::graph::Graph;

fn main() {
    let d = 256i64;
    let dh = 64i64;
    let s = 128i64;

    let mut models: Vec<Model> = vec![
        Model { name: "identity (matmul only)".into(), dims: vec![s, d], ops: vec![] },
        Model {
            name: "split-heads + transpose".into(),
            dims: vec![s, d],
            ops: vec![Op::Split(1, dh), Op::Permute(vec![1, 0, 2])],
        },
    ];
    for layers in [1usize, 2, 4, 8] {
        models.push(deep_round_trip(vec![s, d], dh, layers));
    }

    let pre = preamble();
    println!("A/B: same viewed-matmul model, two layout theories\n");
    println!(
        "{:<26} | {:>9} {:>9} {:>11} | {:>7} {:>8} {:>6}",
        "model", "copy/mat", "seed-nodes", "egraph-rows", "nodes", "unit-K", "same"
    );
    println!("{}", "-".repeat(96));

    for m in &models {
        let mut g = Graph::new();
        build_viewed_matmul(&mut g, m);
        let (copies, mats, _) = backend_copies(&g);
        let (seed, seed_nodes) = affine_seed(m);
        let rows = affine_saturate(&pre, &seed);
        let cute = cute_lower(m);
        let same = cute_verified(m);
        println!(
            "{:<26} | {:>9} {:>9} {:>11} | {:>7} {:>8} {:>6}",
            m.name,
            copies + mats,
            seed_nodes,
            rows,
            cute.nodes(),
            cute.contraction_unit_stride(),
            same
        );
    }
}
