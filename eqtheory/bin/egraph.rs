use cute_layout::algebra::mapping::{coalesce, composition};
use cute_layout::layout::Layout;
use cute_layout::tuple::IntTuple::{self, Leaf, Tuple};
use egglog::EGraph;
use egglog::sort::S;

#[derive(Clone, Copy)]
enum Ruleset {
    Affine,
    Cute,
}

fn ser_tuple(t: &IntTuple) -> String {
    match t {
        Leaf(n) => n.to_string(),
        Tuple(v) => format!("({})", v.iter().map(ser_tuple).collect::<Vec<_>>().join(" ")),
    }
}

fn ser_layout(l: &Layout) -> String {
    format!("{}|{}", ser_tuple(&l.shape), ser_tuple(&l.stride))
}

fn parse_tuple(s: &str) -> (IntTuple, &str) {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix('(') {
        let mut rest = rest;
        let mut items = Vec::new();
        loop {
            rest = rest.trim_start();
            if let Some(after) = rest.strip_prefix(')') {
                return (Tuple(items), after);
            }
            let (t, r) = parse_tuple(rest);
            items.push(t);
            rest = r;
        }
    } else {
        let end = s
            .find(|c: char| c == '(' || c == ')' || c == ' ')
            .unwrap_or(s.len());
        (Leaf(s[..end].parse().unwrap()), &s[end..])
    }
}

fn parse_layout(s: &str) -> Layout {
    let (shape_s, stride_s) = s.split_once('|').unwrap();
    Layout::new(parse_tuple(shape_s).0, parse_tuple(stride_s).0)
}

fn compose_layouts(a: &str, b: &str) -> Option<String> {
    let c = composition(&parse_layout(a), &parse_layout(b)).ok()?;
    Some(ser_layout(&coalesce(&c)))
}

fn new_egraph() -> EGraph {
    let mut eg = EGraph::default();
    egglog::add_primitive!(&mut eg, "cute-compose" = |a: S, b: S| -?> S {
        compose_layouts(a.as_str(), b.as_str()).map(|s| s.into())
    });
    eg
}

const DATATYPES: &str = r#"
(datatype LT (CLit String) (Compose LT LT))
"#;

const CUTE_COMPOSE: &str = r#"
(ruleset cute)
(rewrite (Compose (CLit ?a) (CLit ?b)) (CLit (cute-compose ?a ?b)) :ruleset cute)
"#;

fn round_layout(a: i64, b: i64) -> Layout {
    Layout::new(Tuple(vec![Leaf(a), Leaf(b)]), Tuple(vec![Leaf(b), Leaf(1)]))
}

fn chain_term(factors: &[i64], n: i64) -> String {
    let mut term = format!("(CLit \"{}\")", ser_layout(&round_layout(factors[0], n / factors[0])));
    for &a in &factors[1..] {
        let lit = format!("(CLit \"{}\")", ser_layout(&round_layout(a, n / a)));
        term = format!("(Compose {term} {lit})");
    }
    term
}

fn run(ruleset: Ruleset, factors: &[i64], n: i64) -> String {
    let mut eg = new_egraph();
    let mut program = String::from(DATATYPES);
    if let Ruleset::Cute = ruleset {
        program.push_str(CUTE_COMPOSE);
    }
    program.push_str(&format!("(let t {})\n", chain_term(factors, n)));
    if let Ruleset::Cute = ruleset {
        program.push_str("(run-schedule (saturate cute))\n");
    }
    program.push_str("(extract t)\n");
    eg.parse_and_run_program(None, &program)
        .unwrap()
        .iter()
        .map(|o| o.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn composition_is_address_preserving(factors: &[i64], n: i64) -> bool {
    let mut lay = Layout::new(Leaf(n), Leaf(1));
    for &a in factors {
        lay = composition(&round_layout(a, n / a), &lay).unwrap();
    }
    let direct = |x: i64| {
        let mut y = x;
        for &a in factors.iter().rev() {
            let b = n / a;
            y = (y % a) * b + y / a;
        }
        y
    };
    (0..n).all(|x| lay.call(&Leaf(x)) == direct(x))
}

fn main() {
    let n = 4096i64;
    let factors = [2i64, 4, 8, 16, 32, 64, 128, 2, 4, 8, 16, 32];
    let verified = composition_is_address_preserving(&factors, n);
    println!("cute-compose primitive address-preserving (checked vs cute-layout): {verified}");
    println!();
    println!("ViT-style view chain as e-graph Compose terms; ruleset is a compile option");
    println!(
        "{:>6} | {:>26} | {:>24}",
        "rounds", "affine Compose nodes", "cute Compose nodes"
    );
    println!("{}", "-".repeat(62));
    for k in [1usize, 2, 4, 8, 12] {
        let f = &factors[..k];
        let affine = run(Ruleset::Affine, f, n).matches("Compose").count();
        let cute = run(Ruleset::Cute, f, n).matches("Compose").count();
        println!("{k:>6} | {affine:>26} | {cute:>24}");
    }
}
