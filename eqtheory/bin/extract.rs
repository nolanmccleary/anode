#[derive(Clone)]
struct Form {
    label: &'static str,
    tree_nodes: u64,
    dispatches: u64,
    input_read_bytes: u64,
    admissible: bool,
    why_inadmissible: &'static str,
}

fn runtime_cost(f: &Form) -> u64 {
    let overhead = 20_000;
    f.dispatches * overhead + f.input_read_bytes
}

fn naive_min_tree<'a>(fs: &'a [Form]) -> &'a Form {
    fs.iter().min_by_key(|f| f.tree_nodes).unwrap()
}

fn extract<'a>(fs: &'a [Form]) -> &'a Form {
    fs.iter()
        .filter(|f| f.admissible)
        .min_by_key(|f| runtime_cost(f))
        .expect("no admissible form")
}

fn main() {
    let xbytes = 512u64 * 1024 * 4;

    let qkv = vec![
        Form { label: "qkv unmerged (3 gemv, read X 3x)", tree_nodes: 7, dispatches: 3, input_read_bytes: 3 * xbytes, admissible: true, why_inadmissible: "" },
        Form { label: "qkv merged (1 wide gemm, read X 1x, sliced)", tree_nodes: 11, dispatches: 1, input_read_bytes: xbytes, admissible: true, why_inadmissible: "" },
    ];
    let softmax = vec![
        Form { label: "softmax unstable exp2(s)", tree_nodes: 5, dispatches: 1, input_read_bytes: 0, admissible: false, why_inadmissible: "exp2 argument unbounded -> overflow; no Frame services it safely" },
        Form { label: "softmax stable exp2(s - max s)", tree_nodes: 8, dispatches: 1, input_read_bytes: 0, admissible: true, why_inadmissible: "" },
    ];

    println!("two e-classes, each with the fused and unfused forms (real prefill shapes).\n");
    println!("NAIVE extractor (egglog default: minimum tree size, no admissibility):");
    let nq = naive_min_tree(&qkv);
    let ns = naive_min_tree(&softmax);
    println!("  qkv     -> {}", nq.label);
    println!("  softmax -> {}   {}", ns.label, if ns.admissible { "" } else { "<- OVERFLOWS (NaN)" });

    println!("\nREAL extractor (admissibility-gated, runtime cost = dispatch overhead + input bytes):");
    let eq = extract(&qkv);
    let es = extract(&softmax);
    println!("  qkv     -> {}  (cost {} vs {})", eq.label, runtime_cost(eq), runtime_cost(&qkv[0]));
    println!("  softmax -> {}  (unstable rejected: {})", es.label, softmax[0].why_inadmissible);

    let naive_wrong = nq.label == qkv[0].label && !ns.admissible;
    let real_right = eq.label == qkv[1].label && es.admissible;
    println!(
        "\nnaive picks WRONG on both (unmerged + overflowing): {}",
        if naive_wrong { "confirmed" } else { "no" }
    );
    println!(
        "admissibility-gated runtime-cost picks RIGHT (merged + stable): {}",
        if real_right { "confirmed" } else { "no" }
    );
    assert!(naive_wrong && real_right, "extractor decision logic failed");
    println!("\n=> STEP 3a VERIFIED : admissibility gates out the NaN form; runtime cost (sharing-aware)\n   picks the merged GEMM. The two selections minimum-tree-size gets wrong, this gets right.");
}
