use eqtheory::cute_fuse::find_cogrouped_contractions;
use eqtheory::ir::NodeKind;
use eqtheory::native_model::{build, NativeCfg};

fn main() {
    let cfg = NativeCfg {
        vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
        layers: 1, cap: 4, ctx: 4, seq: 4, prev: 0,
    };
    let nm = build(&cfg, (0..4).collect());
    let g = &nm.g;

    let weight_name = |id: usize| -> String {
        match &g.nodes[id].kind {
            NodeKind::Move { b: Some(b), .. } => match &g.nodes[b.src].kind {
                NodeKind::Input { name } => name.clone(),
                _ => "?".into(),
            },
            _ => "?".into(),
        }
    };

    println!("universal recognition: contractions sharing left operand + extent, constant weights.");
    println!("fires for ANY co-grouped contractions, not a qkv matcher.\n");
    let groups = find_cogrouped_contractions(g);
    for (i, grp) in groups.iter().enumerate() {
        let ws: Vec<String> = grp.iter().map(|&id| weight_name(id)).collect();
        println!("  group {i}: nodes {grp:?} share LHS+K -> mergeable weights {ws:?}");
    }

    let has_qkv = groups.iter().any(|grp| {
        let ws: Vec<String> = grp.iter().map(|&id| weight_name(id)).collect();
        ws.iter().any(|w| w.contains("q_proj")) && ws.iter().any(|w| w.contains("k_proj")) && ws.iter().any(|w| w.contains("v_proj"))
    });
    let has_gate_up = groups.iter().any(|grp| {
        let ws: Vec<String> = grp.iter().map(|&id| weight_name(id)).collect();
        ws.iter().any(|w| w.contains("gate_proj")) && ws.iter().any(|w| w.contains("up_proj"))
    });

    println!("\nfound the QKV group (q+k+v share normed): {}", if has_qkv { "yes" } else { "NO" });
    println!("found the FFN gate+up group (same universal rule): {}", if has_gate_up { "yes" } else { "NO" });
    assert!(has_qkv && has_gate_up, "recognition failed");
    println!("\n=> STEP 3b (recognition) VERIFIED : one universal rule finds BOTH qkv and gate/up.\n   No qkv-specific matcher. Next: synthesize the wide contraction + sliced reads.");
}
