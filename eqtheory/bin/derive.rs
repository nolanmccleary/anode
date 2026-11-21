use egglog::EGraph;

fn main() {
    let program = r#"
(datatype Val
  (In String)
  (One)
  (Mul Val Val)
  (Sub Val Val)
  (Max Val)
  (Sum Val)
  (Pt String Val))
(rewrite (Pt "exp2" ?s) (Mul (Pt "exp2" (Sub ?s (Max ?s))) (Pt "exp2" (Max ?s))))
(birewrite (Sum (Mul ?f ?c)) (Mul (Sum ?f) ?c))
(birewrite (Pt "recip" (Mul ?a ?b)) (Mul (Pt "recip" ?a) (Pt "recip" ?b)))
(rewrite (Mul ?a (Pt "recip" ?a)) (One))
(rewrite (Mul (Pt "recip" ?a) ?a) (One))
(rewrite (Mul ?a (One)) ?a)
(rewrite (Mul (One) ?a) ?a)
(birewrite (Mul ?a ?b) (Mul ?b ?a))
(birewrite (Mul (Mul ?a ?b) ?c) (Mul ?a (Mul ?b ?c)))
(let s (In "s"))
(let plain  (Mul (Pt "exp2" s) (Pt "recip" (Sum (Pt "exp2" s)))))
(let stable (Mul (Pt "exp2" (Sub s (Max s))) (Pt "recip" (Sum (Pt "exp2" (Sub s (Max s)))))))
(run 6)
(check (= plain stable))
"#;
    println!("the mapped graph's softmax is the UNSTABILIZED form (what NaNs):");
    println!("  plain = exp2(s) * recip(Sum(exp2(s)))         -- exp2 of large s overflows");
    println!("the flash kernel computes the stabilized form:");
    println!("  stable = exp2(s-max s) * recip(Sum(exp2(s-max s)))   -- exponent <= 0, bounded\n");
    println!("universal laws only (exp2 homomorphism incl max-factorization, recip algebra,");
    println!("Mul commutative monoid, distribute over Sum). no attn/softmax rule.");
    println!("checking the e-graph DERIVES stable from plain in the real vocabulary:\n");
    let mut eg = EGraph::default();
    match eg.parse_and_run_program(None, program) {
        Ok(_) => println!("=> STABILIZATION DERIVED in real vocabulary : plain == stable.\n   The NaN-free flash form is reached from the mapped unstabilized softmax by universal laws."),
        Err(e) => {
            println!("=> NOT derived: {e}");
            std::process::exit(1);
        }
    }
}
