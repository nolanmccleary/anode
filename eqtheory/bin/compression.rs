use cute_layout::algebra::mapping::{coalesce, composition};
use cute_layout::layout::Layout;
use cute_layout::tuple::IntTuple::{self, Leaf, Tuple};

#[derive(Clone)]
enum E {
    X,
    Lit(i64),
    Add(Box<E>, Box<E>),
    Mul(Box<E>, Box<E>),
    Div(Box<E>, i64),
    Rem(Box<E>, i64),
}

impl E {
    fn eval(&self, x: i64) -> i64 {
        match self {
            E::X => x,
            E::Lit(v) => *v,
            E::Add(a, b) => a.eval(x) + b.eval(x),
            E::Mul(a, b) => a.eval(x) * b.eval(x),
            E::Div(a, d) => a.eval(x).div_euclid(*d),
            E::Rem(a, d) => a.eval(x).rem_euclid(*d),
        }
    }

    fn nodes(&self) -> usize {
        match self {
            E::X | E::Lit(_) => 1,
            E::Add(a, b) | E::Mul(a, b) => 1 + a.nodes() + b.nodes(),
            E::Div(a, _) | E::Rem(a, _) => 1 + a.nodes(),
        }
    }

    fn subst(&self, g: &E) -> E {
        match self {
            E::X => g.clone(),
            E::Lit(v) => E::Lit(*v),
            E::Add(a, b) => E::Add(Box::new(a.subst(g)), Box::new(b.subst(g))),
            E::Mul(a, b) => E::Mul(Box::new(a.subst(g)), Box::new(b.subst(g))),
            E::Div(a, d) => E::Div(Box::new(a.subst(g)), *d),
            E::Rem(a, d) => E::Rem(Box::new(a.subst(g)), *d),
        }
    }
}

fn round_expr(a: i64, b: i64) -> E {
    E::Add(
        Box::new(E::Mul(Box::new(E::Rem(Box::new(E::X), a)), Box::new(E::Lit(b)))),
        Box::new(E::Div(Box::new(E::X), a)),
    )
}

fn round_layout(a: i64, b: i64) -> Layout {
    Layout::new(Tuple(vec![Leaf(a), Leaf(b)]), Tuple(vec![Leaf(b), Leaf(1)]))
}

fn tuple_nodes(t: &IntTuple) -> usize {
    match t {
        Leaf(_) => 1,
        Tuple(v) => 1 + v.iter().map(tuple_nodes).sum::<usize>(),
    }
}

fn main() {
    let n: i64 = 4096;
    let factors = [2i64, 4, 8, 16, 32, 64, 128, 2, 4, 8, 16, 32];
    println!("ViT-style flatten/transpose view chain, N={n}");
    println!("each round reshapes the vector to [a,b] and transposes it (a stride permutation)");
    println!();
    println!(
        "{:>6} | {:>24} | {:>22} | {:>9}",
        "rounds", "substitution term nodes", "cute layout nodes", "same map?"
    );
    println!("{}", "-".repeat(72));
    for k in 1..=factors.len() {
        let mut g = E::X;
        let mut lay = Layout::new(Leaf(n), Leaf(1));
        for &a in factors.iter().take(k) {
            let b = n / a;
            g = round_expr(a, b).subst(&g);
            lay = composition(&round_layout(a, b), &lay).unwrap();
        }
        let lay = coalesce(&lay);
        let subst_nodes = g.nodes();
        let cute_nodes = tuple_nodes(&lay.shape) + tuple_nodes(&lay.stride);
        let same = (0..n).all(|x| g.eval(x) == lay.call(&Leaf(x)));
        println!("{k:>6} | {subst_nodes:>24} | {cute_nodes:>22} | {same:>9}");
    }
}
