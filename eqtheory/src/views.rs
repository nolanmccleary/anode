use luminal::dtype::DType;
use luminal::egglog_snippet::{assembled_program_for, new_egraph};
use luminal::frontend::movement::ViewChain;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex};
use luminal::shape::IntExpr;
use luminal::bufferize::BufferNode;
use luminal_metal::{CompileOptions, HostBuffer, MetalRuntime};

#[derive(Clone)]
pub enum Op {
    Split(usize, i64),
    Permute(Vec<usize>),
    Merge(usize, usize),
}

#[derive(Clone)]
pub struct Model {
    pub name: String,
    pub dims: Vec<i64>,
    pub ops: Vec<Op>,
}

pub fn deep_round_trip(dims: Vec<i64>, inner: i64, layers: usize) -> Model {
    let mut ops = Vec::new();
    for _ in 0..layers {
        ops.push(Op::Split(1, inner));
        ops.push(Op::Permute(vec![1, 0, 2]));
        ops.push(Op::Permute(vec![1, 0, 2]));
        ops.push(Op::Merge(1, 2));
    }
    Model { name: format!("deep round-trip x{layers}"), dims, ops }
}

pub fn apply_view(v: ViewChain, op: &Op) -> ViewChain {
    match op {
        Op::Split(axis, inner) => v.split_dims(*axis, IntExpr::from(*inner)),
        Op::Permute(p) => v.permute(p.as_slice()),
        Op::Merge(a, b) => v.merge_dims(*a, *b),
    }
}

pub fn build_viewed_matmul(g: &mut Graph, m: &Model) -> GraphTensor {
    let shape: Vec<IntExpr> = m.dims.iter().map(|&d| IntExpr::from(d)).collect();
    let x = g.tensor(shape.as_slice(), DType::F32);
    let mut v = x.view();
    for op in &m.ops {
        v = apply_view(v, op);
    }
    let viewed = v.finish();
    let k = *forward_shapes(m).last().unwrap().last().unwrap();
    let w = g.tensor([IntExpr::from(k), IntExpr::from(k)], DType::F32);
    viewed.matmul(w)
}

pub fn backend_copies(g: &Graph) -> (usize, usize, usize) {
    let mut rt = MetalRuntime::load(g).unwrap();
    let inputs: FxHashMap<NodeIndex, HostBuffer> = g
        .logical
        .input_specs()
        .iter()
        .map(|spec| {
            let n: usize = spec.dims.iter().map(|e| e.to_usize().unwrap()).product();
            (spec.id, vec![0f32; n].into())
        })
        .collect();
    let opts = CompileOptions {
        generations: 3,
        generation_size: 4,
        trials: 2,
        seed: 0,
        search_log: false,
        ..CompileOptions::default()
    };
    rt.search(&Default::default(), &Default::default(), &inputs, &opts)
        .unwrap();
    let plan = rt.plan().unwrap();
    let (mut copies, mut materializes, mut computes) = (0usize, 0usize, 0usize);
    for node in plan.dag.node_weights() {
        match node {
            BufferNode::BufferCopy { .. } | BufferNode::BufferCopyValue { .. } => copies += 1,
            BufferNode::Compute { op, .. } => {
                computes += 1;
                let label = op.label();
                if label.contains("Materialize") || label.contains("Copy") {
                    materializes += 1;
                }
            }
            _ => {}
        }
    }
    (copies, materializes, computes)
}

pub fn affine_seed(m: &Model) -> (String, usize) {
    let mut g = Graph::new();
    let shape: Vec<IntExpr> = m.dims.iter().map(|&d| IntExpr::from(d)).collect();
    let mut v = g.tensor(shape.as_slice(), DType::F32).view();
    for op in &m.ops {
        v = apply_view(v, op);
    }
    let viewed = v.finish();
    let seed = g.logical.render(&[viewed.id]).unwrap();
    let nodes = ["IntAdd", "IntMul", "IntTruncDiv", "IntTruncRem", "CoordVar", "IntLit"]
        .iter()
        .map(|t| seed.matches(t).count())
        .sum();
    (seed, nodes)
}

pub fn affine_saturate(preamble: &str, seed: &str) -> usize {
    let mut eg = new_egraph();
    eg.parse_and_run_program(None, preamble).unwrap();
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

pub fn preamble() -> String {
    assembled_program_for(&[])
}

#[derive(Clone)]
pub struct Modes {
    pub modes: Vec<(i64, i64)>,
}

impl Modes {
    pub fn row_major(dims: &[i64]) -> Self {
        let mut stride = 1i64;
        let mut modes = vec![(0i64, 0i64); dims.len()];
        for i in (0..dims.len()).rev() {
            modes[i] = (dims[i], stride);
            stride *= dims[i];
        }
        Modes { modes }
    }
    pub fn apply(mut self, op: &Op) -> Self {
        match op {
            Op::Split(axis, inner) => {
                let (e, s) = self.modes[*axis];
                self.modes.splice(*axis..=*axis, [(e / inner, inner * s), (*inner, s)]);
                self
            }
            Op::Permute(p) => Modes { modes: p.iter().map(|&i| self.modes[i]).collect() },
            Op::Merge(a, b) => {
                let (e0, s0) = self.modes[*a];
                let (e1, s1) = self.modes[*b];
                assert_eq!(s0, e1 * s1, "cute merge requires contiguous modes");
                self.modes.splice(*a..=*b, [(e0 * e1, s1)]);
                self
            }
        }
    }
    pub fn coalesce(mut self) -> Self {
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
    pub fn crd2idx(&self, coords: &[i64]) -> i64 {
        self.modes.iter().zip(coords).map(|(&(_, s), &c)| c * s).sum()
    }
    pub fn dims(&self) -> Vec<i64> {
        self.modes.iter().map(|&(e, _)| e).collect()
    }
    pub fn nodes(&self) -> usize {
        self.modes.len() * 2
    }
    pub fn contraction_unit_stride(&self) -> bool {
        self.modes.last().map(|&(_, s)| s == 1).unwrap_or(false)
    }
}

pub fn cute_lower(m: &Model) -> Modes {
    let mut modes = Modes::row_major(&m.dims);
    for op in &m.ops {
        modes = modes.apply(op);
    }
    modes.coalesce()
}

pub fn forward_shapes(m: &Model) -> Vec<Vec<i64>> {
    let mut shapes = vec![m.dims.clone()];
    for op in &m.ops {
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
                let mm = nd[*a] * nd[*b];
                nd.splice(*a..=*b, [mm]);
                nd
            }
        };
        shapes.push(nd);
    }
    shapes
}

pub fn reference_address(m: &Model, final_coords: &[i64]) -> i64 {
    let shapes = forward_shapes(m);
    let mut cur = final_coords.to_vec();
    for (i, op) in m.ops.iter().enumerate().rev() {
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
                let mm = cur[*a];
                let d_b = in_dims[*b];
                cur.splice(*a..=*a, [mm / d_b, mm % d_b]);
            }
        }
    }
    Modes::row_major(&m.dims).crd2idx(&cur)
}

pub fn decompose(x: i64, shape: &[i64]) -> Vec<i64> {
    let mut coords = vec![0i64; shape.len()];
    let mut rem = x;
    for i in (0..shape.len()).rev() {
        coords[i] = rem % shape[i];
        rem /= shape[i];
    }
    coords
}

pub fn cute_verified(m: &Model) -> bool {
    let cute = cute_lower(m);
    let final_shape = forward_shapes(m).last().unwrap().clone();
    let total: i64 = m.dims.iter().product();
    (0..total.min(4096)).all(|x| {
        cute.crd2idx(&decompose(x, &cute.dims()))
            == reference_address(m, &decompose(x, &final_shape))
    })
}
