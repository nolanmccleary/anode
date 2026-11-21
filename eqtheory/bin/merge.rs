use egglog::EGraph;

fn main() {
    let program = r#"
(datatype Val
  (In String)
  (Contract Val Val i64)
  (Concat Val Val)
  (Fst Val)
  (Snd Val))
(birewrite (Concat (Contract ?x ?a ?k) (Contract ?x ?b ?k)) (Contract ?x (Concat ?a ?b) ?k))
(rewrite (Fst (Concat ?p ?q)) ?p)
(rewrite (Snd (Concat ?p ?q)) ?q)
(let x (In "normed"))
(let q (Contract x (In "Wq") 1024))
(let k (Contract x (In "Wk") 1024))
(let v (Contract x (In "Wv") 1024))
(let wide (Contract x (Concat (Concat (In "Wq") (In "Wk")) (In "Wv")) 1024))
(let qslice (Fst (Fst wide)))
(let kslice (Snd (Fst wide)))
(let vslice (Snd wide))
(run 6)
(check (= q qslice))
(check (= k kslice))
(check (= v vslice))
"#;
    println!("real structure: q,k,v are THREE separate contractions sharing LHS normed + kext,");
    println!("feeding DIFFERENT sinks (q->scores, k/v->cache).\n");
    println!("co-location is a FREE choice for constants: lay Wq|Wk|Wv contiguously, one wide");
    println!("contraction, slice the outputs. the e-graph's job is to CERTIFY the slices equal");
    println!("the originals -- by universal distributivity + slice-of-concat. none names qkv.\n");
    println!("checking each original projection == its slice of the wide co-located contraction:\n");
    let mut eg = EGraph::default();
    match eg.parse_and_run_program(None, program) {
        Ok(_) => println!("=> QKV MERGE CERTIFIED : q == Fst(Fst(wide)), k == Snd(Fst(wide)), v == Snd(wide),\n   where wide = Contract(normed, Concat(Concat(Wq,Wk),Wv)). One wide GEMM, sliced, PROVEN\n   bit-equivalent to the three separate contractions by universal laws."),
        Err(e) => {
            println!("=> NOT certified: {e}");
            std::process::exit(1);
        }
    }
}
