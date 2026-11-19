use eqtheory::cute_egraph::saturate;
use std::collections::{BTreeMap, BTreeSet};

fn main() {
    let data = "(256,256):(256,1)";
    let specs = ["0|16", "1|16", "0.1|4", "1.1|4"];

    println!("BY-MODE TILING under egglog saturation.");
    println!("spec = path|tile: tile the mode at that nested path, cute does the rest.");
    println!("the engine explores every order; congruence merges orders that converge.\n");
    println!("data {data}, specs {specs:?}\n");

    let b = saturate(data, &specs, 4);
    for x in &b {
        println!("  {:<6} {:<36} -> {}", x.spec, x.source, x.access);
    }

    let distinct: BTreeSet<&String> = b.iter().map(|x| &x.access).collect();
    println!(
        "\n{} bindings, {} distinct layouts derived.",
        b.len(),
        distinct.len()
    );

    let mut by_access: BTreeMap<&String, BTreeSet<&String>> = BTreeMap::new();
    for x in &b {
        by_access.entry(&x.access).or_default().insert(&x.source);
    }
    println!("\nconvergence (one layout reached from several sources = one e-class, merged):");
    for (access, sources) in by_access.iter().filter(|(_, s)| s.len() > 1) {
        println!("  {access}");
        for s in sources {
            println!("      from {s}");
        }
    }
}
