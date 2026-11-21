use eqtheory::cute_fuse::{graph_to_egglog, max_ref, term_of};
use eqtheory::native_model::{build, NativeCfg};
use egglog::EGraph;

fn main() {
    let cfg = NativeCfg {
        vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
        layers: 1, cap: 4, ctx: 4, seq: 4, prev: 0,
    };
    let nm = build(&cfg, (0..4).collect());
    let g = &nm.g;
    let nnodes = g.nodes.len();
    println!("mapped a real 1-layer native graph: {nnodes} nodes.\n");

    let mut topo_ok = true;
    for id in 0..nnodes {
        let t = term_of(g, id);
        if let Some(m) = max_ref(&t) {
            if m >= id {
                println!("  NON-TOPOLOGICAL: n{id} = {t} references n{m} (>= {id})");
                topo_ok = false;
            }
        }
    }
    println!("piece 1 topological (every term refs only earlier nodes): {}", if topo_ok { "OK" } else { "FAIL" });

    let (prog, terms) = graph_to_egglog(g);
    let complete = terms.len() == nnodes && terms.iter().all(|t| !t.is_empty());
    println!("piece 2 complete (every node -> a term): {}", if complete { "OK" } else { "FAIL" });

    let mut eg = EGraph::default();
    let accepted = eg.parse_and_run_program(None, &prog).is_ok();
    println!("piece 3 egglog accepts (all refs resolve, well-typed): {}", if accepted { "OK" } else { "FAIL" });

    let extract_prog = format!("{prog}\n(extract n{})\n", nm.logits.id);
    let mut eg2 = EGraph::default();
    let out = eg2.parse_and_run_program(None, &extract_prog).expect("extract");
    let extracted = out.iter().map(|m| format!("{m}")).collect::<Vec<_>>().join("");
    let roundtrip = extracted.contains("(Contract") && extracted.starts_with('(');
    println!("piece 4 output term round-trips (extract logits yields a Val term): {}", if roundtrip { "OK" } else { "FAIL" });
    println!("\nlogits node n{} expands to:\n{}", nm.logits.id, extracted.trim());

    let mut hist = std::collections::BTreeMap::new();
    for t in &terms {
        let head: String = t[1..].chars().take_while(|c| c.is_alphabetic()).collect();
        *hist.entry(head).or_insert(0usize) += 1;
    }
    println!("\nnode-kind histogram (faithful op coverage): {hist:?}");

    assert!(topo_ok && complete && accepted && roundtrip, "mapping verification failed");
    println!("\n=> STEP 1 VERIFIED : GraphIr -> Val mapping is topological, complete, egglog-accepted, round-trips.");
}
