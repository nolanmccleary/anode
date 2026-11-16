use luminal::dtype::DType;
use luminal::egglog_snippet::{assembled_program_for, new_egraph};
use luminal::graph::Graph;
use luminal::shape::IntExpr;

#[derive(Clone)]
enum Op {
    Split(usize, i64),
    Permute(Vec<usize>),
    Merge(usize, usize),
}

fn affine_seed(ops: &[Op], dims: &[i64]) -> (String, usize) {
    let mut g = Graph::new();
    let shape: Vec<IntExpr> = dims.iter().map(|&d| IntExpr::from(d)).collect();
    let mut vc = g.tensor(shape.as_slice(), DType::F32).view();
    for op in ops {
        vc = match op {
            Op::Split(axis, inner) => vc.split_dims(*axis, IntExpr::from(*inner)),
            Op::Permute(p) => vc.permute(p.as_slice()),
            Op::Merge(a, b) => vc.merge_dims(*a, *b),
        };
    }
    let v = vc.finish();
    let seed = g.logical.render(&[v.id]).unwrap();
    let term_nodes = ["IntAdd", "IntMul", "IntTruncDiv", "IntTruncRem", "CoordVar", "IntLit"]
        .iter()
        .map(|t| seed.matches(t).count())
        .sum();
    (seed, term_nodes)
}

fn affine_egraph_rows(seed: &str) -> usize {
    let mut eg = new_egraph();
    eg.parse_and_run_program(None, &assembled_program_for(&[])).unwrap();
    eg.parse_and_run_program(None, seed).unwrap();
    eg.parse_and_run_program(None, "(run-schedule (saturate (run prop)))").unwrap();
    for _ in 0..16 {
        eg.parse_and_run_program(
            None,
            "(run-schedule (run main_ruleset) (run prop) (run subst-walk))",
        )
        .unwrap();
    }
    eg.num_tuples()
}

#[derive(Clone)]
struct Modes {
    modes: Vec<(i64, i64)>,
}

impl Modes {
    fn row_major(dims: &[i64]) -> Self {
        let mut stride = 1i64;
        let mut modes = vec![(0i64, 0i64); dims.len()];
        for i in (0..dims.len()).rev() {
            modes[i] = (dims[i], stride);
            stride *= dims[i];
        }
        Modes { modes }
    }

    fn apply(mut self, op: &Op) -> Self {
        match op {
            Op::Split(axis, inner) => {
                let (e, s) = self.modes[*axis];
                self.modes.splice(*axis..=*axis, [(e / inner, inner * s), (*inner, s)]);
                self
            }
            Op::Permute(p) => Modes {
                modes: p.iter().map(|&i| self.modes[i]).collect(),
            },
            Op::Merge(a, b) => {
                let (e0, s0) = self.modes[*a];
                let (e1, s1) = self.modes[*b];
                assert_eq!(s0, e1 * s1, "cute merge requires contiguous modes");
                self.modes.splice(*a..=*b, [(e0 * e1, s1)]);
                self
            }
        }
    }

    fn coalesce(mut self) -> Self {
        let mut i = 0;
        while i + 1 < self.modes.len() {
            let (e0, s0) = self.modes[i];
            let (e1, s1) = self.modes[i + 1];
            if s0 == e1 * s1 {
                self.modes[i] = (e0 * e1, s1);
                self.modes.remove(i + 1);
            } else {
                i += 1;
            }
        }
        self
    }

    fn crd2idx(&self, coords: &[i64]) -> i64 {
        self.modes.iter().zip(coords).map(|(&(_, s), &c)| c * s).sum()
    }

    fn dims(&self) -> Vec<i64> {
        self.modes.iter().map(|&(e, _)| e).collect()
    }

    fn nodes(&self) -> usize {
        self.modes.len() * 2
    }
}

fn forward_shapes(ops: &[Op], dims: &[i64]) -> Vec<Vec<i64>> {
    let mut shapes = vec![dims.to_vec()];
    for op in ops {
        let d = shapes.last().unwrap().clone();
        let nd = match op {
            Op::Split(axis, inner) => {
                let mut nd = d.clone();
                let e = nd[*axis];
                nd.splice(*axis..=*axis, [e / inner, *inner]);
                nd
            }
            Op::Permute(p) => p.iter().map(|&i| d[i]).collect(),
            Op::Merge(a, b) => {
                let mut nd = d.clone();
                let m = nd[*a] * nd[*b];
                nd.splice(*a..=*b, [m]);
                nd
            }
        };
        shapes.push(nd);
    }
    shapes
}

fn reference_address(ops: &[Op], dims: &[i64], final_coords: &[i64]) -> i64 {
    let shapes = forward_shapes(ops, dims);
    let mut cur = final_coords.to_vec();
    for (i, op) in ops.iter().enumerate().rev() {
        let in_dims = &shapes[i];
        match op {
            Op::Split(axis, inner) => {
                let outer = cur[*axis];
                let inr = cur[*axis + 1];
                cur.splice(*axis..=*axis + 1, [outer * inner + inr]);
            }
            Op::Permute(p) => {
                let mut prev = vec![0i64; p.len()];
                for (new_i, &old_i) in p.iter().enumerate() {
                    prev[old_i] = cur[new_i];
                }
                cur = prev;
            }
            Op::Merge(a, b) => {
                let m = cur[*a];
                let d_b = in_dims[*b];
                cur.splice(*a..=*a, [m / d_b, m % d_b]);
            }
        }
    }
    Modes::row_major(dims).crd2idx(&cur)
}

fn cute_lower(ops: &[Op], dims: &[i64]) -> Modes {
    let mut m = Modes::row_major(dims);
    for op in ops {
        m = m.apply(op);
    }
    m.coalesce()
}

fn decompose(x: i64, shape: &[i64]) -> Vec<i64> {
    let mut coords = vec![0i64; shape.len()];
    let mut rem = x;
    for i in (0..shape.len()).rev() {
        coords[i] = rem % shape[i];
        rem /= shape[i];
    }
    coords
}

fn verify(ops: &[Op], dims: &[i64]) -> bool {
    let cute = cute_lower(ops, dims);
    let final_shape = forward_shapes(ops, dims).last().unwrap().clone();
    let total: i64 = dims.iter().product();
    (0..total.min(4096)).all(|x| {
        cute.crd2idx(&decompose(x, &cute.dims()))
            == reference_address(ops, dims, &decompose(x, &final_shape))
    })
}

fn report(name: &str, ops: &[Op], dims: &[i64]) {
    let (seed, affine_term) = affine_seed(ops, dims);
    let affine_rows = affine_egraph_rows(&seed);
    let cute = cute_lower(ops, dims);
    println!(
        "{name:<34} | affine: {affine_term:>4} term nodes, {affine_rows:>5} egraph rows | cute: {:>3} layout nodes | same map {}",
        cute.nodes(),
        verify(ops, dims)
    );
}

fn main() {
    println!("head-to-head: plausible architecture view ops lowered two ways (affine vs cute)\n");

    let d = 768;
    let dh = 64;
    report(
        "MHA q-proj (split heads+transpose)",
        &[Op::Split(1, dh), Op::Permute(vec![1, 0, 2])],
        &[128, d],
    );
    report(
        "MHA round-trip (heads out, merge)",
        &[
            Op::Split(1, dh),
            Op::Permute(vec![1, 0, 2]),
            Op::Permute(vec![1, 0, 2]),
            Op::Merge(1, 2),
        ],
        &[128, d],
    );

    println!("\nstacked attention reshape/transpose (depth grows, a real multi-layer pattern):");
    for layers in [1usize, 2, 4, 8] {
        let mut ops = Vec::new();
        for _ in 0..layers {
            ops.push(Op::Split(1, dh));
            ops.push(Op::Permute(vec![1, 0, 2]));
            ops.push(Op::Permute(vec![1, 0, 2]));
            ops.push(Op::Merge(1, 2));
        }
        report(&format!("{layers} layers"), &ops, &[128, d]);
    }
}
