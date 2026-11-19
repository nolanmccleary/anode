use eqtheory::cute_egraph::{candidates, extract, saturate, Frame};

fn main() {
    let data = "(40,40):(40,1)";
    let specs = ["0|8", "1|8", "0|16", "1|16"];
    let frames = vec![Frame {
        name: "tiled_matmul",
        rank: 2,
        tiles: vec![vec![8, 8], vec![16, 16]],
    }];

    println!("EXTRACTION over the saturated lattice.");
    println!("frame tiles 8x8/16x16. waste = tiled cells / real cells.");
    println!("tile 8 divides 40 (exact); tile 16 rounds 40->48 (padded, predicated).\n");
    println!("data {data}\n");

    let bindings = saturate(data, &specs, 3);
    let cands = candidates(data, &bindings, &frames);

    println!("admissible candidates in the lattice, ranked:");
    for c in &cands {
        println!(
            "  tile {:?}  waste {:>4.2}x  levels {}  {}",
            c.tile, c.waste, c.levels, c.access
        );
    }

    match extract(data, &bindings, &frames) {
        Some(c) => println!(
            "\nextracted -> tile {:?}, waste {:.2}x: {}",
            c.tile, c.waste, c.access
        ),
        None => println!("\nno admissible tiling"),
    }
}
