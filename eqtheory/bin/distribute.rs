use egglog::EGraph;

fn main() {
    let program = r#"
(datatype Val
  (In String)
  (Contract Val Val)
  (Concat Val Val))
(birewrite (Contract ?x (Concat ?a ?b)) (Concat (Contract ?x ?a) (Contract ?x ?b)))
(let x (In "X"))
(let q (Contract x (In "Wq")))
(let k (Contract x (In "Wk")))
(let v (Contract x (In "Wv")))
(let qkv (Concat (Concat q k) v))
(run 16)
(extract qkv)
"#;
    let mut eg = EGraph::default();
    let msgs = eg
        .parse_and_run_program(None, program)
        .expect("egglog distributivity");
    println!("three contractions X@Wq, X@Wk, X@Wv needed together (qkv = Concat(Concat(q,k),v)).");
    println!("the ONLY rule is distributivity: Contract(x, Concat(a,b)) == Concat(Contract(x,a), Contract(x,b)).");
    println!("no qkv rule, no merge rule. extraction (minimum term) picks:\n");
    for m in &msgs {
        println!("{m}");
    }
    let fused = msgs.iter().any(|m| {
        let c = format!("{m}").replace([' ', '\n'], "");
        c.contains("(Contract(In\"X\")(Concat(Concat(In\"Wq\")(In\"Wk\"))(In\"Wv\")))")
    });
    println!(
        "\n=> {} : the three contractions collapsed to ONE wide contraction over Concat(Wq,Wk,Wv).",
        if fused { "QKV FUSION EMERGED" } else { "NOT fused" }
    );
    assert!(fused, "distributivity did not yield the merged wide contraction");
}
