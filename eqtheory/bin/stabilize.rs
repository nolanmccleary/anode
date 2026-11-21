use egglog::EGraph;

fn main() {
    let program = r#"
(datatype Val
  (In String)
  (Exp Val)
  (Sub Val Val)
  (Div Val Val)
  (Sum Val)
  (Max Val))
(birewrite (Exp (Sub ?a ?b)) (Div (Exp ?a) (Exp ?b)))
(birewrite (Sum (Div ?f ?c)) (Div (Sum ?f) ?c))
(rewrite (Div (Div ?p ?c) (Div ?q ?c)) (Div ?p ?q))
(let x (In "x"))
(let m (Max x))
(let plain  (Div (Exp x)         (Sum (Exp x))))
(let stable (Div (Exp (Sub x m)) (Sum (Exp (Sub x m)))))
(run 30)
(check (= plain stable))
"#;
    let mut eg = EGraph::default();
    println!("softmax(x) = Exp(x)/Sum(Exp(x)) OVERFLOWS for large x (the prefill NaN).");
    println!("flash/online attention computes softmax(x - max x) instead (bounded exponent).");
    println!("the ONLY rules are universal algebra:");
    println!("  exp homomorphism   Exp(a-b) == Exp(a)/Exp(b)");
    println!("  div over sum       Sum(f/c) == Sum(f)/c");
    println!("  cancel common      (p/c)/(q/c) == p/q");
    println!("no attn rule, no softmax rule. checking the e-graph proves them equal:\n");
    match eg.parse_and_run_program(None, program) {
        Ok(_) => println!(
            "=> STABILIZATION EMERGED : softmax(x) == softmax(x - max x) proven from universal laws.\n   The NaN-free form a flash Frame services is in the SAME e-class as plain softmax."
        ),
        Err(e) => {
            println!("=> NOT proven equal: {e}");
            std::process::exit(1);
        }
    }
}
