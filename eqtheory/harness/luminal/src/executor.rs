use crate::cute_probe::{factor, probe};
use eqtheory::metal_backend::CuteMetal;
use cute_layout::algebra::mapping::composition;
use cute_layout::{IntTuple, Layout};
use luminal::dtype::DType;
use luminal::logical_ir::{LogicalOp, MapEntry};
use luminal::prelude::petgraph::stable_graph::StableDiGraph;
use luminal::prelude::petgraph::visit::EdgeRef;
use luminal::prelude::petgraph::{algo::toposort, Direction};
use luminal::prelude::{FxHashMap, NodeIndex};
use metal::Buffer;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

type G = StableDiGraph<luminal::logical_ir::LogicalNode, luminal::logical_ir::InputPort>;

pub struct Val {
    pub buf: Buffer,
    pub dims: Vec<i64>,
    pub dtype: DType,
}

pub struct Exec<'g> {
    pub backend: CuteMetal,
    g: &'g G,
    vals: HashMap<NodeIndex, Val>,
    access: HashMap<NodeIndex, (NodeIndex, Layout)>,
    tape: RefCell<Vec<Tape>>,
    pub dispatches: Cell<usize>,
    pub materializes: Cell<usize>,
    pub inline_reads: Cell<usize>,
    pub matmuls_paired: Cell<usize>,
    pub alloc_total: Cell<usize>,
    mm_skip: std::collections::HashSet<NodeIndex>,
    mm_plan: HashMap<NodeIndex, MatmulPlan>,
    fuse_skip: std::collections::HashSet<NodeIndex>,
    fuse_plan: HashMap<NodeIndex, Fused>,
    pub fused_chains: Cell<usize>,
    pub generic_ew_edges: Cell<usize>,
    pub inplace_cache: bool,
    pub static_idx_folds: Cell<usize>,
    iota_host: HashMap<NodeIndex, Vec<i32>>,
}

fn numel(dims: &[i64]) -> i64 {
    dims.iter().product()
}

fn rm_strides(dims: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        s[k] = s[k + 1] * dims[k + 1];
    }
    s
}

fn colex_strides(dims: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; dims.len()];
    for k in 1..dims.len() {
        s[k] = s[k - 1] * dims[k - 1];
    }
    s
}

fn tuple_of(v: &[i64]) -> IntTuple {
    IntTuple::Tuple(v.iter().map(|x| IntTuple::Leaf(*x)).collect())
}

fn identity_layout(dims: &[i64]) -> Layout {
    Layout::new(tuple_of(dims), tuple_of(&rm_strides(dims)))
}

fn flat_pairs(shape: &IntTuple, stride: &IntTuple, out: &mut Vec<(i64, i64)>) {
    match (shape, stride) {
        (IntTuple::Leaf(s), IntTuple::Leaf(d)) => out.push((*s, *d)),
        (IntTuple::Tuple(ss), IntTuple::Tuple(dd)) => {
            for (a, b) in ss.iter().zip(dd) {
                flat_pairs(a, b, out);
            }
        }
        _ => panic!("ragged layout"),
    }
}

fn meta_of(layout: &Layout, rank: usize) -> Option<[i32; 18]> {
    let mut modes: Vec<(i64, i64)> = Vec::new();
    if rank <= 1 {
        flat_pairs(&layout.shape, &layout.stride, &mut modes);
    } else {
        for ax in (0..rank).rev() {
            let m = layout.get(ax);
            flat_pairs(&m.shape, &m.stride, &mut modes);
        }
    }
    modes.retain(|(s, _)| *s != 1);
    if modes.is_empty() {
        modes.push((1, 0));
    }
    if modes.len() > 8 {
        return None;
    }
    let mut meta = [0i32; 18];
    meta[0] = modes.len() as i32;
    meta[1] = 0;
    for (k, (s, d)) in modes.iter().enumerate() {
        meta[2 + k] = i32::try_from(*s).ok()?;
        meta[10 + k] = i32::try_from(*d).ok()?;
    }
    Some(meta)
}

fn flat_meta(n: i64) -> [i32; 18] {
    let mut m = [0i32; 18];
    m[0] = 1;
    m[2] = n as i32;
    m[10] = 1;
    m
}

fn axis_strides_of(layout: &Layout, rank: usize) -> Option<Vec<i64>> {
    let mut out = Vec::with_capacity(rank);
    for ax in 0..rank {
        let m = if rank <= 1 { layout.clone() } else { layout.get(ax) };
        let mut pairs = Vec::new();
        flat_pairs(&m.shape, &m.stride, &mut pairs);
        pairs.retain(|(sz, _)| *sz != 1);
        match pairs.len() {
            0 => out.push(0),
            1 => out.push(pairs[0].1),
            _ => return None,
        }
    }
    Some(out)
}

fn is_contiguous(layout: &Layout, dims: &[i64]) -> bool {
    meta_of(layout, dims.len()) == meta_of(&identity_layout(dims), dims.len())
}

enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

fn parse_sexpr(s: &str) -> SExpr {
    fn inner(toks: &[String], pos: &mut usize) -> SExpr {
        if toks[*pos] == "(" {
            *pos += 1;
            let mut items = Vec::new();
            while toks[*pos] != ")" {
                items.push(inner(toks, pos));
            }
            *pos += 1;
            SExpr::List(items)
        } else {
            let a = SExpr::Atom(toks[*pos].clone());
            *pos += 1;
            a
        }
    }
    let mut toks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' | ')' => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
                toks.push(ch.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    let mut pos = 0;
    inner(&toks, &mut pos)
}

fn eval_iota(e: &SExpr, coords_from_end: &[i64]) -> i64 {
    let SExpr::List(items) = e else {
        panic!("iota expr leaf outside list");
    };
    let head = match &items[0] {
        SExpr::Atom(a) => a.as_str(),
        _ => panic!("iota expr head"),
    };
    let atom_i64 = |x: &SExpr| -> i64 {
        match x {
            SExpr::Atom(a) => a.parse().expect("iota literal"),
            _ => panic!("iota literal"),
        }
    };
    match head {
        "IntLit" => atom_i64(&items[1]),
        "CoordVar" => {
            let k = atom_i64(&items[2]) as usize;
            coords_from_end[k]
        }
        "IntAdd" => eval_iota(&items[1], coords_from_end) + eval_iota(&items[2], coords_from_end),
        "IntMul" => eval_iota(&items[1], coords_from_end) * eval_iota(&items[2], coords_from_end),
        "IntTruncDiv" => {
            eval_iota(&items[1], coords_from_end) / eval_iota(&items[2], coords_from_end)
        }
        "IntTruncRem" => {
            eval_iota(&items[1], coords_from_end) % eval_iota(&items[2], coords_from_end)
        }
        other => panic!("iota head {other} unsupported — extend the evaluator"),
    }
}

impl<'g> Exec<'g> {
    pub fn new(g: &'g G) -> Self {
        Exec {
            backend: CuteMetal::new(),
            g,
            vals: HashMap::new(),
            access: HashMap::new(),
            tape: RefCell::new(Vec::new()),
            dispatches: Cell::new(0),
            materializes: Cell::new(0),
            inline_reads: Cell::new(0),
            matmuls_paired: Cell::new(0),
            alloc_total: Cell::new(0),
            mm_skip: std::collections::HashSet::new(),
            mm_plan: HashMap::new(),
            fuse_skip: std::collections::HashSet::new(),
            fuse_plan: HashMap::new(),
            fused_chains: Cell::new(0),
            generic_ew_edges: Cell::new(0),
            inplace_cache: false,
            static_idx_folds: Cell::new(0),
            iota_host: HashMap::new(),
        }
    }

    fn alloc(&self, bytes: usize) -> Buffer {
        self.alloc_total.set(self.alloc_total.get() + bytes);
        self.backend.alloc_bytes(bytes)
    }

    fn dims(&self, id: NodeIndex) -> Vec<i64> {
        self.g[id]
            .dims
            .iter()
            .map(|d| d.to_usize().expect("concrete dim") as i64)
            .collect()
    }

    fn operands(&self, id: NodeIndex) -> Vec<NodeIndex> {
        let mut e: Vec<(usize, NodeIndex)> = self
            .g
            .edges_directed(id, Direction::Incoming)
            .map(|edge| {
                let w: &luminal::logical_ir::InputPort = edge.weight();
                (w.0, edge.source())
            })
            .collect();
        e.sort_by_key(|(p, _)| *p);
        e.into_iter().map(|(_, s)| s).collect()
    }

    fn resolve_dry(&mut self, id: NodeIndex) -> Option<(NodeIndex, Layout)> {
        if let Some(a) = self.access.get(&id) {
            return Some(a.clone());
        }
        let a = if matches!(self.g[id].op, LogicalOp::IndexMapApply { .. }) {
            let src = self.operands(id)[0];
            let (base, acc) = self.resolve_dry(src)?;
            let src_dims = self.dims(src);
            let out_dims = self.dims(id);
            let LogicalOp::IndexMapApply { entries } = &self.g[id].op else {
                unreachable!()
            };
            let entries: Vec<MapEntry> = entries.clone();
            let p = probe(&entries, &out_dims, &colex_strides(&src_dims))?;
            if p.offset != 0 {
                return None;
            }
            let link: Layout = p.layout.parse().unwrap();
            (base, composition(&acc, &link).ok()?)
        } else {
            (id, identity_layout(&self.dims(id)))
        };
        self.access.insert(id, a.clone());
        Some(a)
    }

    fn ctor(&self, id: NodeIndex) -> &'static str {
        self.g[id].op.constructor()
    }

    fn sole_consumer(&self, id: NodeIndex) -> Option<NodeIndex> {
        let mut it = self.g.neighbors_directed(id, Direction::Outgoing);
        let c = it.next()?;
        if it.next().is_some() {
            return None;
        }
        Some(c)
    }

    fn const_value(&self, id: NodeIndex) -> Option<f64> {
        let mut cur = id;
        for _ in 0..3 {
            match &self.g[cur].op {
                LogicalOp::Constant(v) | LogicalOp::ConstantF64(v) => return Some(*v),
                LogicalOp::IndexMapApply { .. } => cur = self.operands(cur)[0],
                _ => return None,
            }
        }
        None
    }

    fn fusion_facts(&self) -> String {
        let mut f = String::new();
        for id in self.g.node_indices() {
            let i = id.index() as i64;
            f.push_str(&format!("(node {i} \"{}\")\n", self.ctor(id)));
            let n = self.g.neighbors_directed(id, Direction::Outgoing).count();
            f.push_str(&format!("(consumers {i} {n})\n"));
            for (p, src) in self.operands(id).iter().enumerate() {
                f.push_str(&format!("(operand {i} {p} {})\n", src.index()));
            }
            if self.const_value(id).is_some() {
                f.push_str(&format!("(constb {i})\n"));
            }
            for src in self.operands(id) {
                if self.g[src].dims == self.g[id].dims {
                    f.push_str(&format!("(dimeq {i} {})\n", src.index()));
                }
            }
        }
        f
    }

    fn plan_fusions(&mut self) {
        let program = format!(
            "{}{}{}",
            "(relation node (i64 String))\n(relation operand (i64 i64 i64))\n(relation consumers (i64 i64))\n(relation constb (i64))\n(relation dimeq (i64 i64))\n(relation ew-kind (String))\n(ew-kind \"LogicalMul\")\n(ew-kind \"LogicalAdd\")\n(ew-kind \"LogicalRecip\")\n(ew-kind \"LogicalSqrt\")\n(ew-kind \"LogicalExp2\")\n(relation ew (i64))\n(rule ((node ?n ?c) (ew-kind ?c)) ((ew ?n)))\n(relation chained (i64 i64))\n(rule ((ew ?a) (ew ?b) (operand ?b ?p ?a) (consumers ?a 1) (dimeq ?b ?a)) ((chained ?a ?b)))\n(relation rms (i64 i64 i64))\n(relation silu (i64 i64 i64 i64 i64 i64))\n(relation smax (i64 i64))\n(relation claim (i64))\n(relation constb-view (i64))\n(rule ((constb ?n)) ((constb-view ?n)))\n(rule ((node ?v \"LogicalIndexMapApply\") (operand ?v 0 ?c) (constb-view ?c)) ((constb-view ?v)))\n",
            self.fusion_facts(),
            "(rule ((node ?m2 \"LogicalMul\") (operand ?m2 0 ?x) (operand ?m2 1 ?x) (node ?rs \"LogicalReduceSum\") (operand ?rs 0 ?m2) (consumers ?m2 1) (node ?c1m \"LogicalMul\") (operand ?c1m ?pc ?rs) (consumers ?rs 1) (node ?ae \"LogicalAdd\") (operand ?ae ?pa ?c1m) (consumers ?c1m 1) (operand ?ae ?pe ?ev) (constb-view ?ev) (node ?sq \"LogicalSqrt\") (operand ?sq 0 ?ae) (consumers ?ae 1) (node ?rc \"LogicalRecip\") (operand ?rc 0 ?sq) (consumers ?sq 1) (node ?vw \"LogicalIndexMapApply\") (operand ?vw 0 ?rc) (consumers ?rc 1) (node ?fin \"LogicalMul\") (operand ?fin ?pf ?vw) (consumers ?vw 1) (operand ?fin ?px ?x)) ((rms ?fin ?x ?ev) (claim ?m2) (claim ?rs) (claim ?c1m) (claim ?ae) (claim ?sq) (claim ?rc) (claim ?vw)))\n(rule ((node ?e \"LogicalExp2\") (operand ?e 0 ?m1) (node ?m1 \"LogicalMul\") (operand ?m1 0 ?m0) (node ?m0 \"LogicalMul\") (operand ?m0 0 ?g) (operand ?m0 1 ?c1) (constb-view ?c1) (operand ?m1 1 ?c2) (constb-view ?c2) (node ?ad \"LogicalAdd\") (operand ?ad ?q0 ?e) (consumers ?e 1) (operand ?ad ?q1 ?c3) (constb-view ?c3) (node ?rc \"LogicalRecip\") (operand ?rc 0 ?ad) (consumers ?ad 1) (node ?sg \"LogicalMul\") (operand ?sg ?q2 ?rc) (consumers ?rc 1) (operand ?sg ?q3 ?g) (node ?fin \"LogicalMul\") (operand ?fin ?q4 ?sg) (consumers ?sg 1) (operand ?fin ?q5 ?u) (!= ?u ?sg)) ((silu ?fin ?g ?u ?c1 ?c2 ?c3) (claim ?m0) (claim ?m1) (claim ?e) (claim ?ad) (claim ?rc) (claim ?sg)))\n(rule ((node ?a \"LogicalAdd\") (consumers ?a 2) (node ?rmx \"LogicalReduceMax\") (operand ?rmx 0 ?a) (node ?vmx \"LogicalIndexMapApply\") (operand ?vmx 0 ?rmx) (consumers ?rmx 1) (node ?nm \"LogicalMul\") (operand ?nm ?r0 ?vmx) (consumers ?vmx 1) (operand ?nm ?r1 ?cn) (constb-view ?cn) (node ?a2 \"LogicalAdd\") (operand ?a2 ?r2 ?a) (operand ?a2 ?r3 ?nm) (consumers ?nm 1) (node ?lm \"LogicalMul\") (operand ?lm ?r4 ?a2) (consumers ?a2 1) (operand ?lm ?r5 ?cl) (constb-view ?cl) (node ?e \"LogicalExp2\") (operand ?e 0 ?lm) (consumers ?lm 1) (consumers ?e 2) (node ?rsm \"LogicalReduceSum\") (operand ?rsm 0 ?e) (node ?vsm \"LogicalIndexMapApply\") (operand ?vsm 0 ?rsm) (consumers ?rsm 1) (node ?rcp \"LogicalRecip\") (operand ?rcp 0 ?vsm) (consumers ?vsm 1) (node ?fin \"LogicalMul\") (operand ?fin ?r6 ?e) (operand ?fin ?r7 ?rcp) (consumers ?rcp 1) (!= ?rmx ?rsm)) ((smax ?fin ?a) (claim ?rmx) (claim ?vmx) (claim ?nm) (claim ?a2) (claim ?lm) (claim ?e) (claim ?rsm) (claim ?vsm) (claim ?rcp)))\n(run 4)\n"
        );
        let mut eg = egglog::EGraph::default();
        eg.parse_and_run_program(None, &program).expect("fusion egglog");
        let ni = |v: i64| NodeIndex::new(v as usize);
        let rows = |eg: &mut egglog::EGraph, name: &str, arity: usize| -> Vec<Vec<i64>> {
            let out = eg
                .parse_and_run_program(None, &format!("(print-function {name} 1000000)"))
                .expect("print");
            let mut res = Vec::new();
            for o in out {
                if let egglog::CommandOutput::PrintFunction(_f, dag, rows, _m) = o {
                    for (key, _) in rows {
                        if let egglog::Term::App(nm, children) = dag.get(key) {
                            if nm.as_str() == name && children.len() == arity {
                                let kids = children.clone();
                                res.push(
                                    (0..arity)
                                        .map(|k| dag.to_string(kids[k]).parse::<i64>().unwrap())
                                        .collect(),
                                );
                            }
                        }
                    }
                }
            }
            res
        };
        for r in rows(&mut eg, "rms", 3) {
            let eps = self.const_value(ni(r[2])).expect("rms eps const") as f32;
            self.fuse_plan.insert(ni(r[0]), Fused::Rms { x: ni(r[1]), eps });
        }
        for r in rows(&mut eg, "silu", 6) {
            let c1 = self.const_value(ni(r[3])).expect("silu c1") ;
            let c2 = self.const_value(ni(r[4])).expect("silu c2");
            let c3 = self.const_value(ni(r[5])).expect("silu c3") as f32;
            self.fuse_plan.insert(
                ni(r[0]),
                Fused::SiluMul { g: ni(r[1]), u: ni(r[2]), k: (c1 * c2) as f32, c3 },
            );
        }
        for r in rows(&mut eg, "smax", 2) {
            self.fuse_plan.insert(ni(r[0]), Fused::Softmax { scores: ni(r[1]) });
        }
        for r in rows(&mut eg, "claim", 1) {
            self.fuse_skip.insert(ni(r[0]));
        }
        let chains = rows(&mut eg, "chained", 2);
        let claimed: std::collections::HashSet<i64> = rows(&mut eg, "claim", 1)
            .into_iter()
            .map(|r| r[0])
            .collect();
        let free: Vec<(NodeIndex, NodeIndex)> = chains
            .iter()
            .filter(|c| !claimed.contains(&c[0]) && !claimed.contains(&c[1]))
            .map(|c| (ni(c[0]), ni(c[1])))
            .filter(|(a, b)| !self.fuse_plan.contains_key(a) && !self.fuse_plan.contains_key(b))
            .collect();
        self.plan_chains(&free);
        self.plan_attention();
        let generic = free
            .iter()
            .filter(|(a, b)| {
                !self.fuse_skip.contains(a)
                    && !self.fuse_plan.contains_key(a)
                    && !self.fuse_skip.contains(b)
                    && !self.fuse_plan.contains_key(b)
            })
            .count();
        if std::env::var("EDGES").is_ok() {
            for c in chains.iter().filter(|c| !claimed.contains(&c[0]) && !claimed.contains(&c[1])) {
                let (a, b) = (ni(c[0]), ni(c[1]));
                println!(
                    "edge {} {:?} {:?} -> {} {:?} {:?}",
                    c[0], self.ctor(a), self.dims(a), c[1], self.ctor(b), self.dims(b)
                );
            }
        }
        self.generic_ew_edges.set(generic);
    }

    fn chain_step(&self, node: NodeIndex, predecessor: Option<NodeIndex>) -> Option<(i32, i32, i32, Option<NodeIndex>, Option<NodeIndex>)> {
        let op = match self.ctor(node) {
            "LogicalMul" => 0,
            "LogicalAdd" => 1,
            "LogicalRecip" => 2,
            "LogicalSqrt" => 3,
            "LogicalExp2" => 4,
            _ => return None,
        };
        let ops = self.operands(node);
        let rest: Vec<NodeIndex> = ops
            .iter()
            .copied()
            .filter(|o| Some(*o) != predecessor)
            .collect();
        match (op, predecessor, rest.len()) {
            (2..=4, Some(_), 0) => Some((op, 0, 0, None, None)),
            (2..=4, None, 1) => Some((op, 0, 0, None, Some(rest[0]))),
            (0 | 1, Some(_), 1) => {
                let aux = rest[0];
                if let Some(v) = self.const_value(aux) {
                    Some((op, 1, (v as f32).to_bits() as i32, None, None))
                } else if self.g[aux].dims == self.g[node].dims {
                    Some((op, 2, 0, Some(aux), None))
                } else {
                    None
                }
            }
            (0 | 1, None, 2) => {
                let (p, aux) = if self.const_value(rest[0]).is_none()
                    && self.g[rest[0]].dims == self.g[node].dims
                {
                    (rest[0], rest[1])
                } else {
                    (rest[1], rest[0])
                };
                if self.g[p].dims != self.g[node].dims || self.const_value(p).is_some() {
                    return None;
                }
                if let Some(v) = self.const_value(aux) {
                    Some((op, 1, (v as f32).to_bits() as i32, None, Some(p)))
                } else if self.g[aux].dims == self.g[node].dims {
                    Some((op, 2, 0, Some(aux), Some(p)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn plan_chains(&mut self, edges: &[(NodeIndex, NodeIndex)]) {
        let next: HashMap<NodeIndex, NodeIndex> = edges.iter().copied().collect();
        let prevs: std::collections::HashSet<NodeIndex> = edges.iter().map(|(_, b)| *b).collect();
        for (root, _) in edges.iter().filter(|(a, _)| !prevs.contains(a)) {
            let mut nodes = vec![*root];
            let mut cur = *root;
            while let Some(nb) = next.get(&cur) {
                nodes.push(*nb);
                cur = *nb;
            }
            if nodes.len() > 8 {
                continue;
            }
            let mut prog = [0i32; 25];
            let mut exts: Vec<NodeIndex> = Vec::new();
            let mut primary = None;
            let mut ok = true;
            for (i, nd) in nodes.iter().enumerate() {
                let predecessor = if i == 0 { None } else { Some(nodes[i - 1]) };
                let Some((op, kind, payload, ext, prim)) = self.chain_step(*nd, predecessor) else {
                    ok = false;
                    break;
                };
                if i == 0 {
                    match prim {
                        Some(p) => primary = Some(p),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                let mut payload = payload;
                if kind == 2 {
                    let e = ext.unwrap();
                    let idx = exts.iter().position(|x| *x == e).unwrap_or_else(|| {
                        exts.push(e);
                        exts.len() - 1
                    });
                    if idx >= 3 {
                        ok = false;
                        break;
                    }
                    payload = idx as i32 + 1;
                }
                prog[1 + 3 * i] = op;
                prog[2 + 3 * i] = kind;
                prog[3 + 3 * i] = payload;
            }
            if !ok {
                continue;
            }
            prog[0] = nodes.len() as i32;
            let tail = *nodes.last().unwrap();
            for n in &nodes[..nodes.len() - 1] {
                self.fuse_skip.insert(*n);
            }
            self.fuse_plan.insert(
                tail,
                Fused::Chain { primary: primary.unwrap(), exts, prog },
            );
        }
    }

    fn axis_stride(&mut self, id: NodeIndex, axis: usize) -> Option<i64> {
        let (_, acc) = self.resolve_dry(id)?;
        let rank = self.dims(id).len();
        let ms = axis_modes(&acc, axis, rank);
        match ms.len() {
            0 => Some(0),
            1 => Some(ms[0].1),
            _ => None,
        }
    }

    fn plan_attention(&mut self) {
        let smaxes: Vec<(NodeIndex, NodeIndex)> = self
            .fuse_plan
            .iter()
            .filter_map(|(fin, f)| match f {
                Fused::Softmax { scores } => Some((*fin, *scores)),
                _ => None,
            })
            .collect();
        for (fin, scores) in smaxes {
            let Some(Fused::Chain { primary: qk_rs, exts, prog }) =
                self.fuse_plan.get(&scores).cloned()
            else {
                continue;
            };
            if prog[0] != 2 || prog[1] != 0 || prog[2] != 1 || prog[4] != 1 || prog[5] != 2 {
                continue;
            }
            let scale_bits = prog[3];
            let bias = exts[0];
            let LogicalOp::ReduceSum { axis_from_end } = self.g[qk_rs].op else { continue };
            let qk_mul = self.operands(qk_rs)[0];
            if !matches!(self.g[qk_mul].op, LogicalOp::Mul) {
                continue;
            }
            let qk_ops = self.operands(qk_mul);
            let Some(vw) = self.sole_consumer(fin) else { continue };
            if self.ctor(vw) != "LogicalIndexMapApply" {
                continue;
            }
            let Some(pv_mul) = self.sole_consumer(vw) else { continue };
            if self.ctor(pv_mul) != "LogicalMul" {
                continue;
            }
            let Some(pv_rs) = self.sole_consumer(pv_mul) else { continue };
            if !matches!(self.g[pv_rs].op, LogicalOp::ReduceSum { .. }) {
                continue;
            }
            let v = *self.operands(pv_mul).iter().find(|o| **o != vw).unwrap();
            let dims = self.dims(qk_mul);
            let rank = dims.len();
            let hd_ax = rank - 1 - axis_from_end;
            let hdn = dims[hd_ax];
            if hdn > 64 {
                continue;
            }
            let mut kv_ax = None;
            let mut a_only: Vec<usize> = Vec::new();
            let mut b_only: Vec<usize> = Vec::new();
            let mut ok = true;
            for ax in 0..rank {
                if ax == hd_ax || dims[ax] == 1 {
                    continue;
                }
                let (Some(sa), Some(sb)) = (
                    self.axis_stride(qk_ops[0], ax),
                    self.axis_stride(qk_ops[1], ax),
                ) else {
                    ok = false;
                    break;
                };
                match (sa != 0, sb != 0) {
                    (true, true) => {
                        if kv_ax.replace(ax).is_some() {
                            ok = false;
                            break;
                        }
                    }
                    (true, false) => a_only.push(ax),
                    (false, true) => b_only.push(ax),
                    (false, false) => {}
                }
            }
            if !ok {
                continue;
            }
            let out_numel = numel(&self.dims(pv_rs));
            let kvn = kv_ax.map(|ax| dims[ax]).unwrap_or(1);
            let pick = |q_axes: &Vec<usize>, k_axes: &Vec<usize>| -> Option<(Vec<usize>, usize)> {
                if k_axes.len() != 1 || q_axes.is_empty() || q_axes.len() > 2 {
                    return None;
                }
                let prod: i64 = q_axes.iter().map(|a| dims[*a]).product();
                (out_numel == kvn * prod * hdn).then(|| (q_axes.clone(), k_axes[0]))
            };
            let (q_op, k_op, q_axes, ctx_ax) =
                if let Some((qa, ca)) = pick(&a_only, &b_only) {
                    (qk_ops[0], qk_ops[1], qa, ca)
                } else if let Some((qa, ca)) = pick(&b_only, &a_only) {
                    (qk_ops[1], qk_ops[0], qa, ca)
                } else {
                    continue;
                };
            let (g_ax, s_ax) = match q_axes.len() {
                1 => (q_axes[0], None),
                _ => (q_axes[0], Some(q_axes[1])),
            };
            let (gn, sn, ctxn) = (
                dims[g_ax],
                s_ax.map(|a| dims[a]).unwrap_or(1),
                dims[ctx_ax],
            );
            let st = |me: &mut Self, id: NodeIndex, ax: Option<usize>| -> Option<i64> {
                match ax {
                    Some(ax) => me.axis_stride(id, ax),
                    None => Some(0),
                }
            };
            let (Some(qkv), Some(qg), Some(qs), Some(qhd)) = (
                st(self, q_op, kv_ax),
                st(self, q_op, Some(g_ax)),
                st(self, q_op, s_ax),
                st(self, q_op, Some(hd_ax)),
            ) else {
                continue;
            };
            let (Some(kkv), Some(kctx), Some(khd)) = (
                st(self, k_op, kv_ax),
                st(self, k_op, Some(ctx_ax)),
                st(self, k_op, Some(hd_ax)),
            ) else {
                continue;
            };
            let pvm_dims = self.dims(pv_mul);
            let pvr = pvm_dims.len();
            let LogicalOp::ReduceSum { axis_from_end: pv_afe } = self.g[pv_rs].op else {
                continue;
            };
            let pv_ctx_ax = pvr - 1 - pv_afe;
            if pvm_dims[pv_ctx_ax] != ctxn {
                continue;
            }
            let mut pv_hd_ax = None;
            let mut pv_kv_ax = None;
            for ax in 0..pvr {
                if ax == pv_ctx_ax || pvm_dims[ax] == 1 {
                    continue;
                }
                let sv = self.axis_stride(v, ax);
                let sw = self.axis_stride(vw, ax);
                match (sv.map(|x| x != 0), sw.map(|x| x != 0)) {
                    (Some(true), Some(true)) => {
                        if pv_kv_ax.replace(ax).is_some() {
                            ok = false;
                            break;
                        }
                    }
                    (Some(true), Some(false)) => {
                        if pv_hd_ax.replace(ax).is_some() {
                            ok = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if !ok {
                continue;
            }
            let Some(pv_hd_ax) = pv_hd_ax else { continue };
            if pvm_dims[pv_hd_ax] != hdn {
                continue;
            }
            let (Some(vkv), Some(vctx), Some(vhd)) = (
                st(self, v, pv_kv_ax),
                st(self, v, Some(pv_ctx_ax)),
                st(self, v, Some(pv_hd_ax)),
            ) else {
                continue;
            };
            let (Some(bkv), Some(bg), Some(bs), Some(bctx)) = (
                st(self, bias, kv_ax),
                st(self, bias, Some(g_ax)),
                st(self, bias, s_ax),
                st(self, bias, Some(ctx_ax)),
            ) else {
                continue;
            };
            let fits = |x: i64| i32::try_from(x).is_ok();
            if ![qkv, qg, qs, qhd, kkv, kctx, khd, vkv, vctx, vhd, bkv, bg, bs, bctx]
                .iter()
                .all(|x| fits(*x))
            {
                continue;
            }
            let mut params = [0i32; 25];
            params[0] = kvn as i32;
            params[1] = gn as i32;
            params[2] = sn as i32;
            params[3] = ctxn as i32;
            params[4] = hdn as i32;
            params[5] = 0;
            params[6] = qkv as i32;
            params[7] = qg as i32;
            params[8] = qs as i32;
            params[9] = qhd as i32;
            params[10] = 0;
            params[11] = kkv as i32;
            params[12] = kctx as i32;
            params[13] = khd as i32;
            params[14] = 0;
            params[15] = vkv as i32;
            params[16] = vctx as i32;
            params[17] = vhd as i32;
            params[18] = 0;
            params[19] = bkv as i32;
            params[20] = bg as i32;
            params[21] = bs as i32;
            params[22] = bctx as i32;
            params[23] = scale_bits;
            for n in [qk_mul, qk_rs, scores, fin, vw, pv_mul] {
                self.fuse_skip.insert(n);
            }
            self.fuse_plan.remove(&scores);
            self.fuse_plan.remove(&fin);
            self.fuse_plan.insert(
                pv_rs,
                Fused::Attn { q: q_op, k: k_op, v, bias, params },
            );
        }
    }

    // When an operand's strides cannot group to one (row, col) stride pair, the
    // matmul can instead read a row-major copy of the operand's true elements:
    // broadcast (stride-0) axes squeeze to extent 1, so the copy holds m*k (or
    // k*n) elements, never the full broadcast product. Batched plans keep
    // original-layout batch strides, so a copy would invalidate them — refuse.
    fn squeeze_plan(&self, l: &Layout, dims: &[i64], rank: usize, batched: bool) -> Option<(Layout, Vec<i64>)> {
        if batched {
            return None;
        }
        let sq: Vec<i64> = (0..rank)
            .map(|ax| if axis_modes(l, ax, rank).iter().any(|(_, d)| *d != 0) { dims[ax] } else { 1 })
            .collect();
        Some((identity_layout(&sq), sq))
    }

    // Dispatch-time companion of squeeze_plan: copy the operand's varying axes
    // (coordinates on squeezed axes pinned at 0) into a dense row-major buffer.
    fn squeezed_copy(&mut self, src: NodeIndex, sq_dims: &[i64]) -> Buffer {
        let (base, acc) = self.resolve(src);
        let (mut ss, mut dd) = (Vec::new(), Vec::new());
        for (ax, sq) in sq_dims.iter().enumerate() {
            if *sq == 1 {
                ss.push(IntTuple::Leaf(1));
                dd.push(IntTuple::Leaf(0));
            } else {
                let m = acc.get(ax);
                let mut flat = Vec::new();
                flat_pairs(&m.shape, &m.stride, &mut flat);
                let extent: i64 = flat.iter().map(|(s, _)| s).product();
                assert_eq!(extent, *sq, "squeezed copy axis extent");
                ss.push(m.shape.clone());
                dd.push(m.stride.clone());
            }
        }
        let sq_acc = Layout::new(IntTuple::Tuple(ss), IntTuple::Tuple(dd));
        let n = numel(sq_dims);
        let meta = meta_of(&sq_acc, sq_dims.len()).expect("squeezed copy meta");
        let out = self.alloc(n as usize * 4);
        let srcb = self.vals[&base].buf.clone();
        self.run_xcopy("xcopy_f", &srcb, &out, &[], n, meta, flat_meta(n), [0; 8], [0; 8]);
        self.materializes.set(self.materializes.get() + 1);
        out
    }

    fn plan_matmuls(&mut self) {
        let dbg = std::env::var("MMDEBUG").is_ok();
        let rej = |rs: NodeIndex, mul: NodeIndex, numel: i64, why: &str| {
            if dbg {
                eprintln!("MMDEBUG reject rs={} mul={} numel={numel}: {why}", rs.index(), mul.index());
            }
        };
        let ids: Vec<NodeIndex> = self.g.node_indices().collect();
        for rs in ids {
            let LogicalOp::ReduceSum { axis_from_end } = self.g[rs].op else {
                continue;
            };
            if self.fuse_skip.contains(&rs) {
                continue;
            }
            let mul = self.operands(rs)[0];
            if !matches!(self.g[mul].op, LogicalOp::Mul) {
                continue;
            }
            let nconsumers = self.g.neighbors_directed(mul, Direction::Outgoing).count();
            let ops = self.operands(mul);
            let dims = self.dims(mul);
            let numel: i64 = dims.iter().product();
            if nconsumers != 1 {
                rej(rs, mul, numel, &format!("mul has {nconsumers} consumers"));
                continue;
            }
            let rank = dims.len();
            let ax = rank - 1 - axis_from_end;
            let Some((a_base, al)) = self.resolve_dry(ops[0]) else {
                rej(rs, mul, numel, "operand a not dry-resolvable");
                continue;
            };
            let Some((b_base, bl)) = self.resolve_dry(ops[1]) else {
                rej(rs, mul, numel, "operand b not dry-resolvable");
                continue;
            };
            let varies = |l: &Layout, axis: usize| {
                axis_modes(l, axis, rank).iter().any(|(_, d)| *d != 0)
            };
            if !varies(&al, ax) || !varies(&bl, ax) {
                rej(rs, mul, numel, "operand constant along contraction axis");
                continue;
            }
            let (mut m_axes, mut n_axes, mut b_axes) = (Vec::new(), Vec::new(), Vec::new());
            let mut ok = true;
            for axis in 0..rank {
                if axis == ax {
                    continue;
                }
                match (varies(&al, axis), varies(&bl, axis)) {
                    (true, true) => b_axes.push(axis),
                    (_, true) => n_axes.push(axis),
                    _ => {
                        if dims[axis] != 1 {
                            m_axes.push(axis)
                        }
                    }
                }
            }
            if b_axes.len() > 4 {
                ok = false;
            }
            if !ok || m_axes.iter().any(|ma| n_axes.iter().any(|na| ma > na)) {
                rej(rs, mul, numel, &format!("axis classification m={m_axes:?} n={n_axes:?} b={b_axes:?}"));
                continue;
            }
            let out_dims_full = self.dims(rs);
            if !b_axes.is_empty() {
                let innermost_b = *b_axes.iter().max().unwrap();
                let mn_min = m_axes.iter().chain(n_axes.iter()).min().copied();
                if let Some(mn) = mn_min {
                    if innermost_b > mn {
                        rej(rs, mul, numel, "batch axis inside m/n axes");
                        continue;
                    }
                }
                assert_eq!(
                    out_dims_full.len() + 1,
                    rank,
                    "reduce drops exactly the contraction axis"
                );
            }
            let mut batch = vec![b_axes.len() as i32];
            let mut nbatch = 1i64;
            let mut bfit = true;
            for axis in &b_axes {
                let (Some(abs_), Some(bbs)) = (
                    group_stride(axis_modes(&al, *axis, rank)),
                    group_stride(axis_modes(&bl, *axis, rank)),
                ) else {
                    bfit = false;
                    break;
                };
                let oax = if *axis < ax { *axis } else { *axis - 1 };
                let cstride: i64 = out_dims_full[oax + 1..].iter().product();
                if [abs_, bbs, cstride, dims[*axis]]
                    .iter()
                    .any(|v| i32::try_from(*v).is_err())
                {
                    bfit = false;
                    break;
                }
                batch.extend([dims[*axis] as i32, abs_ as i32, bbs as i32, cstride as i32]);
                nbatch *= dims[*axis];
            }
            if !bfit {
                rej(rs, mul, numel, "batch strides not groupable or exceed i32");
                continue;
            }
            let collect = |l: &Layout, axes: &[usize]| -> Vec<(i64, i64)> {
                let mut v = Vec::new();
                for axis in axes.iter().rev() {
                    v.extend(axis_modes(l, *axis, rank));
                }
                v
            };
            let m: i64 = m_axes.iter().map(|x| dims[*x]).product();
            let n: i64 = n_axes.iter().map(|x| dims[*x]).product();
            let k = dims[ax];
            // When an operand's strides cannot group to one (row, col) stride pair,
            // plan against a contiguous copy of the operand (value-identical, copied
            // at dispatch via contiguous_val) instead of leaving the node to the
            // naive path, which materializes the full m*k*n broadcast product.
            let (al, a_fix) = match (
                group_stride(collect(&al, &m_axes)),
                group_stride(collect(&al, &[ax])),
            ) {
                (Some(_), Some(_)) => (al, None),
                _ => match self.squeeze_plan(&al, &dims, rank, !b_axes.is_empty()) {
                    Some((l, sq)) => (l, Some((ops[0], sq))),
                    None => {
                        rej(rs, mul, numel, "a strides not groupable, batched copy not plannable");
                        continue;
                    }
                },
            };
            let (bl, b_fix) = match (
                group_stride(collect(&bl, &[ax])),
                group_stride(collect(&bl, &n_axes)),
            ) {
                (Some(_), Some(_)) => (bl, None),
                _ => match self.squeeze_plan(&bl, &dims, rank, !b_axes.is_empty()) {
                    Some((l, sq)) => (l, Some((ops[1], sq))),
                    None => {
                        rej(rs, mul, numel, "b strides not groupable, batched copy not plannable");
                        continue;
                    }
                },
            };
            let Some(ars) = group_stride(collect(&al, &m_axes)) else {
                rej(rs, mul, numel, "a row strides not groupable");
                continue;
            };
            let Some(acs) = group_stride(collect(&al, &[ax])) else {
                rej(rs, mul, numel, "a contraction strides not groupable");
                continue;
            };
            let Some(brs) = group_stride(collect(&bl, &[ax])) else {
                rej(rs, mul, numel, "b contraction strides not groupable");
                continue;
            };
            let Some(bcs) = group_stride(collect(&bl, &n_axes)) else {
                rej(rs, mul, numel, "b col strides not groupable");
                continue;
            };
            let fits = |v: i64| i32::try_from(v).is_ok();
            if ![ars, acs, brs, bcs, m, n, k].iter().all(|v| fits(*v)) {
                rej(rs, mul, numel, "strides or extents exceed i32");
                continue;
            }
            self.mm_skip.insert(mul);
            self.mm_plan.insert(
                rs,
                MatmulPlan {
                    a_base,
                    b_base,
                    a_fix,
                    b_fix,
                    m,
                    k,
                    n,
                    ap: [0, ars as i32, acs as i32, 0],
                    bp: [0, brs as i32, bcs as i32, 0],
                    batch,
                    nbatch,
                },
            );
        }
    }

    fn resolve(&mut self, id: NodeIndex) -> (NodeIndex, Layout) {
        if let Some(a) = self.access.get(&id) {
            return a.clone();
        }
        let a = if matches!(self.g[id].op, LogicalOp::IndexMapApply { .. }) {
            let src = self.operands(id)[0];
            let (base, acc) = self.resolve(src);
            let src_dims = self.dims(src);
            let out_dims = self.dims(id);
            let LogicalOp::IndexMapApply { entries } = &self.g[id].op else {
                unreachable!()
            };
            let entries: Vec<MapEntry> = entries.clone();
            let p = probe(&entries, &out_dims, &colex_strides(&src_dims))
                .expect("view link not probe-representable — refuse");
            if p.offset != 0 {
                let nb = self.materialize_value(src, base, acc);
                let link: Layout = p.layout.parse().unwrap();
                let composed = composition(&identity_layout(&src_dims), &link)
                    .expect("compose against fresh base");
                (nb, composed)
            } else {
                let link: Layout = p.layout.parse().unwrap();
                match composition(&acc, &link) {
                    Ok(c) => (base, c),
                    Err(_) => {
                        let nb = self.materialize_value(src, base, acc);
                        let composed = composition(&identity_layout(&src_dims), &link)
                            .expect("compose against fresh base");
                        (nb, composed)
                    }
                }
            }
        } else {
            (id, identity_layout(&self.dims(id)))
        };
        self.access.insert(id, a.clone());
        a
    }

    fn run_xcopy(
        &self,
        kernel: &'static str,
        src: &Buffer,
        dst: &Buffer,
        idx: &[(Buffer, [i32; 18])],
        n: i64,
        rmeta: [i32; 18],
        wmeta: [i32; 18],
        rspec: [i32; 8],
        wspec: [i32; 8],
    ) {
        assert!(idx.len() <= 4, "xcopy index tensor cap");
        let d = |k: usize| idx.get(k).map(|(b, _)| b.clone()).unwrap_or_else(|| src.clone());
        let (i0, i1, i2, i3) = (d(0), d(1), d(2), d(3));
        let im = |k: usize| idx.get(k).map(|(_, m)| *m).unwrap_or([0i32; 18]);
        self.run1d(
            kernel,
            &[src, dst, &i0, &i1, &i2, &i3],
            &[
                (6, Scalar(n as u32)),
                (7, Meta(rmeta)),
                (8, Meta(wmeta)),
                (9, Spec(rspec)),
                (10, Spec(wspec)),
                (11, Meta(im(0))),
                (12, Meta(im(1))),
                (13, Meta(im(2))),
                (14, Meta(im(3))),
            ],
            n as u64,
        );
    }

    fn materialize_value(&mut self, id: NodeIndex, base: NodeIndex, acc: Layout) -> NodeIndex {
        if self.vals.contains_key(&id) {
            return id;
        }
        let dims = self.dims(id);
        let n = numel(&dims);
        let dtype = self.g[id].dtype;
        let meta = meta_of(&acc, dims.len()).expect("materialize meta");
        let kernel = match dtype {
            DType::F32 => "xcopy_f",
            DType::Int => "xcopy_i",
            other => panic!("materialize dtype {other:?}"),
        };
        let bytes = n as usize * 4;
        let out = self.alloc(bytes);
        let src = self.vals[&base].buf.clone();
        self.run_xcopy(kernel, &src, &out, &[], n, meta, flat_meta(n), [0; 8], [0; 8]);
        self.materializes.set(self.materializes.get() + 1);
        self.vals.insert(id, Val { buf: out, dims, dtype });
        self.access.insert(id, (id, identity_layout(&self.dims(id))));
        id
    }

    fn meta_walk(meta: &[i32; 18], i: i64) -> i64 {
        let nm = meta[0] as i64;
        let mut off = meta[1] as i64;
        let mut r = i;
        for k in 0..nm as usize {
            let sz = meta[2 + k] as i64;
            off += (r % sz) * meta[10 + k] as i64;
            r /= sz;
        }
        off
    }

    fn static_index_fold(&mut self, coord: NodeIndex, st: i64, n: i64) -> Option<(i64, Vec<(i64, i64)>)> {
        let (base, acc) = self.resolve(coord);
        let host = self.iota_host.get(&base)?.clone();
        let meta = meta_of(&acc, self.dims(coord).len())?;
        let w: Vec<i64> = (0..n)
            .map(|gid| host[Self::meta_walk(&meta, gid) as usize] as i64)
            .collect();
        let off = w[0];
        let modes = factor(&w)?;
        for gid in 0..n {
            let mut v = off;
            let mut r = gid;
            for (sz, d) in &modes {
                v += (r % sz) * d;
                r /= sz;
            }
            if v != w[gid as usize] {
                return None;
            }
        }
        let scaled: Vec<(i64, i64)> = modes.iter().map(|(sz, d)| (*sz, d * st)).collect();
        if scaled.iter().any(|(sz, d)| i32::try_from(*sz).is_err() || i32::try_from(*d).is_err()) {
            return None;
        }
        i32::try_from(off * st).ok()?;
        Some((off * st, scaled))
    }

    fn merge_meta(meta: &mut [i32; 18], off_add: i64, modes: &[(i64, i64)]) -> bool {
        let nm = meta[0] as usize;
        if nm + modes.len() > 8 {
            return false;
        }
        for (k, (sz, d)) in modes.iter().enumerate() {
            meta[2 + nm + k] = *sz as i32;
            meta[10 + nm + k] = *d as i32;
        }
        meta[0] = (nm + modes.len()) as i32;
        meta[1] += off_add as i32;
        true
    }

    fn index_operand(&mut self, id: NodeIndex) -> (Buffer, [i32; 18]) {
        let (base, meta) = self.operand_meta(id);
        (self.vals[&base].buf.clone(), meta)
    }

    fn contiguous_val(&mut self, id: NodeIndex) -> NodeIndex {
        let (base, acc) = self.resolve(id);
        let dims = self.dims(id);
        if base == id && is_contiguous(&acc, &dims) {
            return id;
        }
        if is_contiguous(&acc, &dims) && self.dims(base) == dims {
            return base;
        }
        self.materialize_value(id, base, acc)
    }

    fn run1d(&self, kernel: &'static str, bufs: &[&Buffer], args: &[(u64, Arg)], n: u64) {
        const MAX_DISPATCH: u64 = 1 << 27;
        assert!(
            n <= MAX_DISPATCH,
            "dispatch of {kernel} with n={n} exceeds {MAX_DISPATCH} threads — a planner \
             miss is materializing a giant intermediate; refusing to pin that much memory"
        );
        self.tape.borrow_mut().push(Tape::Run1d {
            kernel,
            bufs: bufs.iter().map(|b| (*b).clone()).collect(),
            args: args.to_vec(),
            n,
        });
        self.dispatches.set(self.dispatches.get() + 1);
    }

    pub fn traffic_bytes(&self) -> usize {
        self.tape
            .borrow()
            .iter()
            .map(|t| match t {
                Tape::Run1d { kernel, n, args, .. } => {
                    let n = *n as usize;
                    match *kernel {
                        "add_strided" | "mul_strided" => 3 * n * 4,
                        "copy_strided" | "copy_strided_i" | "recip_o" | "sqrt_o" | "exp2_o"
                        | "cast_i2f" => 2 * n * 4,
                        "cast_b2f" => n + n * 4,
                        "less_than_i" => 2 * n * 4 + n,
                        "reduce_sum_axis" | "reduce_max_axis" => n * 4,
                        "rmsnorm_row" => 2 * n * 4,
                        "xcopy_f" | "xcopy_i" | "xcopy_i2f" => 3 * n * 4,
                        "lt_strided" => 2 * n * 4 + n,
                        "reduce_sum_l" | "reduce_max_l" => {
                            let axis = match args.get(1).map(|a| &a.1) {
                                Some(Scalar(v)) => *v as usize,
                                _ => 1,
                            };
                            n * axis * 4 + n * 4
                        }
                        "silu_mul" => 3 * n * 4,
                        "ew_chain" => 3 * n * 4,
                        "attn_decode" | "attn_online" => 4 * n,
                        "softmax_rows" => 2 * n * 4,
                        other => panic!("traffic for {other}"),
                    }
                }
                Tape::MetaOnly { n, .. } => {
                    let n = *n as usize;
                    n * 4 * 2 + n * 4 * 2
                }
                Tape::Matmul { m, k, n, nbatch, .. } => {
                    (*nbatch as usize) * (m * k + k * n + m * n) * 4
                }
            })
            .sum()
    }

    pub fn replay(&self) {
        if std::env::var("ONEENC").is_ok() {
            return self.replay_one_encoder();
        }
        if std::env::var("TAPETIME").is_ok() {
            return self.replay_tape_timed();
        }
        let cb = self.backend.owned_command_buffer();
        for t in self.tape.borrow().iter() {
            self.encode_tape_entry(&cb, t);
        }
        cb.commit();
        cb.wait_until_completed();
    }

    fn replay_tape_timed(&self) {
        let mut totals: std::collections::HashMap<String, (f64, usize)> = Default::default();
        for t in self.tape.borrow().iter() {
            let cb = self.backend.owned_command_buffer();
            let t0 = std::time::Instant::now();
            self.encode_tape_entry(&cb, t);
            cb.commit();
            cb.wait_until_completed();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let label = match t {
                Tape::Run1d { kernel, n, .. } => format!("{kernel} n={n}"),
                Tape::MetaOnly { kernel, n, .. } => format!("{kernel} n={n}"),
                Tape::Matmul { m, k, n, nbatch, .. } => format!("matmul {m}x{k}x{n} b={nbatch}"),
            };
            let e = totals.entry(label).or_insert((0.0, 0));
            e.0 += ms;
            e.1 += 1;
        }
        let mut v: Vec<_> = totals.into_iter().collect();
        v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        for (label, (ms, count)) in v.iter().take(20) {
            eprintln!("TAPETIME {ms:>10.1}ms x{count:<5} {label}");
        }
    }

    fn encode_tape_entry(&self, cb: &metal::CommandBufferRef, t: &Tape) {
        {
            match t {
                Tape::Run1d { kernel, bufs, args, n } => {
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(self.backend.pipeline(kernel));
                    for (k, b) in bufs.iter().enumerate() {
                        enc.set_buffer(k as u64, Some(b), 0);
                    }
                    for (slot, m) in args {
                        match m {
                            Meta(meta) => enc.set_bytes(*slot, 72, meta.as_ptr() as *const _),
                            Scalar(v) => enc.set_bytes(*slot, 4, v as *const u32 as *const _),
                            F32(v) => enc.set_bytes(*slot, 4, v as *const f32 as *const _),
                            Prog(p) => enc.set_bytes(*slot, 100, p.as_ptr() as *const _),
                            Spec(p) => enc.set_bytes(*slot, 32, p.as_ptr() as *const _),
                        }
                    }
                    let tg = 256u64.min((*n).max(1));
                    enc.dispatch_threads(
                        metal::MTLSize::new((*n).max(1), 1, 1),
                        metal::MTLSize::new(tg, 1, 1),
                    );
                    enc.end_encoding();
                }
                Tape::MetaOnly { kernel, bufs, meta, n } => {
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(self.backend.pipeline(kernel));
                    for (k, b) in bufs.iter().enumerate() {
                        enc.set_buffer(k as u64, Some(b), 0);
                    }
                    enc.set_bytes(6, 72, meta.as_ptr() as *const _);
                    let tg = 256u64.min((*n).max(1));
                    enc.dispatch_threads(
                        metal::MTLSize::new((*n).max(1), 1, 1),
                        metal::MTLSize::new(tg, 1, 1),
                    );
                    enc.end_encoding();
                }
                Tape::Matmul { a, b, c, m, k, n, ap, bp, cp, batch, nbatch } => {
                    self.backend.contract_into(cb, a, b, c, *m, *k, *n, *ap, *bp, *cp, batch, *nbatch);
                }
            }
        }
    }

    fn operand_meta(&mut self, id: NodeIndex) -> (NodeIndex, [i32; 18]) {
        let (base, acc) = self.resolve(id);
        let dims = self.dims(id);
        match meta_of(&acc, dims.len()) {
            Some(m) => {
                if base != id {
                    self.inline_reads.set(self.inline_reads.get() + 1);
                }
                (base, m)
            }
            None => {
                let nb = self.materialize_value(id, base, acc);
                let m = meta_of(&identity_layout(&dims), dims.len()).unwrap();
                (nb, m)
            }
        }
    }

    fn dispatch_fused(&mut self, id: NodeIndex, f: Fused) {
        let dims = self.dims(id);
        let n = numel(&dims);
        match f {
            Fused::Rms { x, eps } => {
                let xv = self.contiguous_val(x);
                let cols = *dims.last().unwrap();
                let rows = n / cols;
                let out = self.alloc(n as usize * 4);
                let xb = &self.vals[&xv].buf;
                self.run1d(
                    "rmsnorm_row",
                    &[xb, &out],
                    &[
                        (2, Scalar(rows as u32)),
                        (3, Scalar(cols as u32)),
                        (4, F32(eps)),
                    ],
                    rows as u64 * 256,
                );
                self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
            }
            Fused::SiluMul { g, u, k, c3 } => {
                let gv = self.contiguous_val(g);
                let uv = self.contiguous_val(u);
                let out = self.alloc(n as usize * 4);
                let (gb, ub) = (&self.vals[&gv].buf, &self.vals[&uv].buf);
                self.run1d(
                    "silu_mul",
                    &[gb, ub, &out],
                    &[(3, Scalar(n as u32)), (4, F32(k)), (5, F32(c3))],
                    n as u64,
                );
                self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
            }
            Fused::Attn { q, k, v, bias, params } => {
                let (qb, _) = self.operand_meta(q);
                let (kb, _) = self.operand_meta(k);
                let (vb, _) = self.operand_meta(v);
                let (bb, _) = self.operand_meta(bias);
                let jobs = (params[0] * params[1] * params[2]) as u64;
                let out = self.alloc(n as usize * 4);
                let bufs: Vec<Buffer> = [&qb, &kb, &vb, &bb]
                    .iter()
                    .map(|x| self.vals[*x].buf.clone())
                    .collect();
                let threads = jobs.div_ceil(8) * 256;
                self.run1d(
                    "attn_online",
                    &[&bufs[0], &bufs[1], &bufs[2], &bufs[3], &out],
                    &[(5, Prog(params))],
                    threads,
                );
                self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
            }
            Fused::Chain { primary, exts, prog } => {
                let (pb, pm) = self.operand_meta(primary);
                let mut ebufs: Vec<(NodeIndex, [i32; 18])> = Vec::new();
                for e in &exts {
                    ebufs.push(self.operand_meta(*e));
                }
                let out = self.alloc(n as usize * 4);
                let zero = meta_of(&identity_layout(&[1]), 1).unwrap();
                let b0 = self.vals[&pb].buf.clone();
                let get = |k: usize| -> metal::Buffer {
                    ebufs
                        .get(k)
                        .map(|(id, _)| self.vals[id].buf.clone())
                        .unwrap_or_else(|| b0.clone())
                };
                let (b1, b2, b3) = (get(0), get(1), get(2));
                let m = |k: usize| ebufs.get(k).map(|(_, m)| *m).unwrap_or(zero);
                self.run1d(
                    "ew_chain",
                    &[&b0, &b1, &b2, &b3, &out],
                    &[
                        (5, Scalar(n as u32)),
                        (6, Prog(prog)),
                        (7, Meta(pm)),
                        (8, Meta(m(0))),
                        (9, Meta(m(1))),
                        (10, Meta(m(2))),
                    ],
                    n as u64,
                );
                self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
            }
            Fused::Softmax { scores } => {
                let sv = self.contiguous_val(scores);
                let cols = *dims.last().unwrap();
                let rows = n / cols;
                let sb = self.vals[&sv].buf.clone();
                self.run1d(
                    "softmax_rows",
                    &[&sb],
                    &[(1, Scalar(rows as u32)), (2, Scalar(cols as u32))],
                    rows as u64,
                );
                self.vals.insert(id, Val { buf: sb, dims, dtype: DType::F32 });
            }
        }
        self.fused_chains.set(self.fused_chains.get() + 1);
    }

    pub fn run(&mut self, inputs: &FxHashMap<NodeIndex, luminal_metal::HostBuffer>) {
        if std::env::var("CHAINS").is_ok() {
            let order = toposort(&self.g, None).unwrap();
            for id in order.iter() {
                let outs = self.g.neighbors_directed(*id, Direction::Outgoing).count();
                let ops: Vec<String> = self
                    .operands(*id)
                    .iter()
                    .map(|o| format!("{}", o.index()))
                    .collect();
                println!(
                    "{:>4} {:<24} dims {:?} consumers {} operands [{}]",
                    id.index(),
                    self.g[*id].op.constructor(),
                    self.dims(*id),
                    outs,
                    ops.join(",")
                );
            }
        }
        if std::env::var("GENERAL_ONLY").is_err() {
            self.plan_fusions();
        }
        self.plan_matmuls();
        if std::env::var("PLANONLY").is_ok() {
            return;
        }
        let order = toposort(&self.g, None).expect("logical graph is a DAG");
        for id in order {
            if self.mm_skip.contains(&id) || self.fuse_skip.contains(&id) {
                continue;
            }
            if let Some(f) = self.fuse_plan.get(&id).cloned() {
                self.dispatch_fused(id, f);
                continue;
            }
            if let Some(plan) = self.mm_plan.get(&id).cloned() {
                let dims = self.dims(id);
                let n_out = numel(&dims);
                assert_eq!(n_out, plan.m * plan.n * plan.nbatch, "matmul out extents");
                let a_buf = match &plan.a_fix {
                    Some((src, sq)) => self.squeezed_copy(*src, sq),
                    None => self.vals[&plan.a_base].buf.clone(),
                };
                let b_buf = match &plan.b_fix {
                    Some((src, sq)) => self.squeezed_copy(*src, sq),
                    None => self.vals[&plan.b_base].buf.clone(),
                };
                let out = self.alloc(n_out as usize * 4);
                let _ = self.backend.search_cached(&eqtheory::kernelset::Composition::Contraction {
                    shape: eqtheory::native_exec::ContractShape {
                        m: plan.m as usize,
                        n: plan.n as usize,
                        ap: plan.ap,
                        bp: plan.bp,
                        cp: [0, plan.n as i32, 1, 0],
                        batch: plan.batch.clone(),
                        nbatch: plan.nbatch,
                    },
                    kext: plan.k as i64,
                });
                self.tape.borrow_mut().push(Tape::Matmul {
                    a: a_buf,
                    b: b_buf,
                    c: out.clone(),
                    m: plan.m as usize,
                    k: plan.k as usize,
                    n: plan.n as usize,
                    ap: plan.ap,
                    bp: plan.bp,
                    cp: [0, plan.n as i32, 1, 0],
                    batch: plan.batch.clone(),
                    nbatch: plan.nbatch,
                });
                self.dispatches.set(self.dispatches.get() + 1);
                self.matmuls_paired.set(self.matmuls_paired.get() + 1);
                self.inline_reads.set(self.inline_reads.get() + 2);
                self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                continue;
            }
            let dims = self.dims(id);
            let n = numel(&dims);
            match &self.g[id].op {
                LogicalOp::Input { .. } => {
                    let hb = inputs.get(&id).expect("input buffer");
                    let (buf, dtype) = match self.g[id].dtype {
                        DType::F32 => (self.backend.upload(&hb.as_f32().unwrap()), DType::F32),
                        DType::Int => (self.backend.upload_i32(&hb.as_i32().unwrap()), DType::Int),
                        other => panic!("input dtype {other:?}"),
                    };
                    self.vals.insert(id, Val { buf, dims, dtype });
                }
                LogicalOp::Constant(v) | LogicalOp::ConstantF64(v) => {
                    let buf = self.backend.upload(&vec![*v as f32; n as usize]);
                    self.vals.insert(id, Val { buf, dims, dtype: DType::F32 });
                }
                LogicalOp::Iota { value_expr } => {
                    let e = parse_sexpr(value_expr);
                    let rank = dims.len();
                    let mut host = Vec::with_capacity(n as usize);
                    for i in 0..n {
                        let mut r = i;
                        let mut from_end = vec![0i64; rank];
                        for ax in (0..rank).rev() {
                            from_end[rank - 1 - ax] = r % dims[ax];
                            r /= dims[ax];
                        }
                        host.push(eval_iota(&e, &from_end) as i32);
                    }
                    let buf = self.backend.upload_i32(&host);
                    self.iota_host.insert(id, host);
                    self.vals.insert(id, Val { buf, dims, dtype: DType::Int });
                }
                LogicalOp::Cast(dt) | LogicalOp::RoundingCast(dt) => {
                    assert_eq!(*dt, DType::F32, "only casts to F32 appear in this model");
                    let src = self.operands(id)[0];
                    let (base, _) = self.resolve(src);
                    match self.vals[&base].dtype {
                        DType::Int => {
                            let (sb, meta) = self.index_operand(src);
                            let out = self.alloc(n as usize * 4);
                            self.run_xcopy(
                                "xcopy_i2f",
                                &sb,
                                &out,
                                &[],
                                n,
                                meta,
                                flat_meta(n),
                                [0; 8],
                                [0; 8],
                            );
                            self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                        }
                        DType::Bool => {
                            let sv = self.contiguous_val(src);
                            let out = self.alloc(n as usize * 4);
                            let sb = &self.vals[&sv].buf;
                            self.run1d("cast_b2f", &[sb, &out], &[(2, Scalar(n as u32))], n as u64);
                            self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                        }
                        other => panic!("cast from {other:?}"),
                    }
                }
                LogicalOp::Sqrt | LogicalOp::Exp2 | LogicalOp::Recip => {
                    let kernel = match self.g[id].op {
                        LogicalOp::Sqrt => "sqrt_o",
                        LogicalOp::Exp2 => "exp2_o",
                        _ => "recip_o",
                    };
                    let src = self.operands(id)[0];
                    let sv = self.contiguous_val(src);
                    let out = self.alloc(n as usize * 4);
                    let sb = &self.vals[&sv].buf;
                    self.run1d(kernel, &[sb, &out], &[(2, Scalar(n as u32))], n as u64);
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                }
                LogicalOp::Add | LogicalOp::Mul => {
                    let kernel = if matches!(self.g[id].op, LogicalOp::Add) {
                        "add_strided"
                    } else {
                        "mul_strided"
                    };
                    let ops = self.operands(id);
                    let (ab, am) = self.operand_meta(ops[0]);
                    let (bb, bm) = self.operand_meta(ops[1]);
                    let out = self.alloc(n as usize * 4);
                    let (abuf, bbuf) = (&self.vals[&ab].buf, &self.vals[&bb].buf);
                    self.run1d(kernel, &[abuf, bbuf, &out], &[(3, Scalar(n as u32)), (4, Meta(am)), (5, Meta(bm))], n as u64);
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                }
                LogicalOp::LessThan => {
                    let ops = self.operands(id);
                    let (ab, am) = self.operand_meta(ops[0]);
                    let (bb, bm) = self.operand_meta(ops[1]);
                    let out = self.alloc(n as usize);
                    let (abuf, bbuf) = (&self.vals[&ab].buf, &self.vals[&bb].buf);
                    self.run1d(
                        "lt_strided",
                        &[abuf, bbuf, &out],
                        &[(3, Scalar(n as u32)), (4, Meta(am)), (5, Meta(bm))],
                        n as u64,
                    );
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::Bool });
                }
                LogicalOp::ReduceSum { axis_from_end } | LogicalOp::ReduceMax { axis_from_end } => {
                    let kernel = if matches!(self.g[id].op, LogicalOp::ReduceSum { .. }) {
                        "reduce_sum_l"
                    } else {
                        "reduce_max_l"
                    };
                    let src = self.operands(id)[0];
                    let (sb, meta) = self.index_operand(src);
                    let sdims = self.dims(src);
                    let axis = sdims.len() - 1 - axis_from_end;
                    let outer: i64 = sdims[..axis].iter().product();
                    let axlen = sdims[axis];
                    let inner: i64 = sdims[axis + 1..].iter().product();
                    assert_eq!(outer * inner, n, "reduce extents");
                    let out = self.alloc(n as usize * 4);
                    self.run1d(
                        kernel,
                        &[&sb, &out],
                        &[
                            (2, Scalar(outer as u32)),
                            (3, Scalar(axlen as u32)),
                            (4, Scalar(inner as u32)),
                            (5, Meta(meta)),
                        ],
                        n as u64,
                    );
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                }
                LogicalOp::Gather => {
                    let ops = self.operands(id);
                    let data = ops[0];
                    let (dbase, dacc) = self.resolve(data);
                    let ddims = self.dims(data);
                    let rank = ddims.len();
                    assert!(rank <= 4, "gather rank cap");
                    assert_eq!(ops.len(), rank + 1, "gather operands = data + rank coords");
                    let (dbuf, dstrides) = match axis_strides_of(&dacc, rank) {
                        Some(st) => (self.vals[&dbase].buf.clone(), st),
                        None => {
                            let dv = self.materialize_value(data, dbase, dacc);
                            (self.vals[&dv].buf.clone(), rm_strides(&ddims))
                        }
                    };
                    let mut rmeta = [0i32; 18];
                    let mut idx: Vec<(Buffer, [i32; 18])> = Vec::new();
                    let mut rspec = [0i32; 8];
                    for (k, st) in dstrides.iter().enumerate() {
                        let coord = ops[1 + k];
                        let folded = self
                            .static_index_fold(coord, *st, n)
                            .and_then(|(off, modes)| {
                                Self::merge_meta(&mut rmeta, off, &modes).then_some(())
                            });
                        match folded {
                            Some(()) => {
                                self.static_idx_folds.set(self.static_idx_folds.get() + 1)
                            }
                            None => {
                                let nd = rspec[0] as usize;
                                let io = self.index_operand(coord);
                                idx.push(io);
                                rspec[1 + 2 * nd] = (idx.len() - 1) as i32;
                                rspec[2 + 2 * nd] = *st as i32;
                                rspec[0] += 1;
                            }
                        }
                    }
                    let out = self.alloc(n as usize * 4);
                    self.run_xcopy(
                        "xcopy_f",
                        &dbuf,
                        &out,
                        &idx,
                        n,
                        rmeta,
                        flat_meta(n),
                        rspec,
                        [0; 8],
                    );
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                }
                LogicalOp::Scatter => {
                    let ops = self.operands(id);
                    let (ibase, iacc) = self.resolve(ops[0]);
                    let idims = self.dims(ops[0]);
                    assert_eq!(idims, dims, "scatter init matches out");
                    let imeta = meta_of(&iacc, idims.len()).expect("scatter init meta");
                    let alias = self.inplace_cache
                        && ibase == ops[0]
                        && matches!(self.g[ibase].op, LogicalOp::Input { .. });
                    let out = if alias {
                        self.vals[&ibase].buf.clone()
                    } else {
                        self.alloc(n as usize * 4)
                    };
                    let ibuf = self.vals[&ibase].buf.clone();
                    if !alias {
                        self.run_xcopy("xcopy_f", &ibuf, &out, &[], n, imeta, flat_meta(n), [0; 8], [0; 8]);
                    }
                    let src = *ops.last().unwrap();
                    let (sbase, smeta) = self.operand_meta(src);
                    let sdims = self.dims(src);
                    let src_n = numel(&sdims);
                    let rank = dims.len();
                    assert!(rank <= 4, "scatter rank cap");
                    assert_eq!(ops.len(), rank + 2, "scatter operands = init + coords + src");
    let strides = rm_strides(&dims);
                    let mut wmeta = [0i32; 18];
                    let mut idx: Vec<(Buffer, [i32; 18])> = Vec::new();
                    let mut wspec = [0i32; 8];
                    for (k, st) in strides.iter().enumerate() {
                        let coord = ops[1 + k];
                        let folded = self
                            .static_index_fold(coord, *st, src_n)
                            .and_then(|(off, modes)| {
                                Self::merge_meta(&mut wmeta, off, &modes).then_some(())
                            });
                        match folded {
                            Some(()) => {
                                self.static_idx_folds.set(self.static_idx_folds.get() + 1)
                            }
                            None => {
                                let nd = wspec[0] as usize;
                                let io = self.index_operand(coord);
                                idx.push(io);
                                wspec[1 + 2 * nd] = (idx.len() - 1) as i32;
                                wspec[2 + 2 * nd] = *st as i32;
                                wspec[0] += 1;
                            }
                        }
                    }
                    let sbuf = self.vals[&sbase].buf.clone();
                    self.run_xcopy(
                        "xcopy_f",
                        &sbuf,
                        &out,
                        &idx,
                        src_n,
                        smeta,
                        wmeta,
                        [0; 8],
                        wspec,
                    );
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                }
                LogicalOp::IndexMapApply { .. } => {}
                other => panic!(
                    "constructor {} not handled by the cute executor",
                    other.constructor()
                ),
            }
        }
    }

    fn replay_one_encoder(&self) {
        let cb = self.backend.owned_command_buffer();
        let tape = self.tape.borrow();
        let mut i = 0usize;
        while i < tape.len() {
            match &tape[i] {
                Tape::Matmul { a, b, c, m, k, n, ap, bp, cp, batch, nbatch } => {
                    self.backend.contract_into(&cb, a, b, c, *m, *k, *n, *ap, *bp, *cp, batch, *nbatch);
                    i += 1;
                }
                _ => {
                    let enc = cb.new_compute_command_encoder();
                    while i < tape.len() {
                        let written: &Buffer = match &tape[i] {
                            Tape::Run1d { kernel, bufs, args, n } => {
                                enc.set_compute_pipeline_state(self.backend.pipeline(kernel));
                                for (kk, bb) in bufs.iter().enumerate() {
                                    enc.set_buffer(kk as u64, Some(bb), 0);
                                }
                                for (slot, m) in args {
                                    match m {
                                        Meta(meta) => enc.set_bytes(*slot, 72, meta.as_ptr() as *const _),
                                        Scalar(v) => enc.set_bytes(*slot, 4, v as *const u32 as *const _),
                                        F32(v) => enc.set_bytes(*slot, 4, v as *const f32 as *const _),
                                        Prog(p) => enc.set_bytes(*slot, 100, p.as_ptr() as *const _),
                                        Spec(p) => enc.set_bytes(*slot, 32, p.as_ptr() as *const _),
                                    }
                                }
                                let tg = 256u64.min((*n).max(1));
                                enc.dispatch_threads(
                                    metal::MTLSize::new((*n).max(1), 1, 1),
                                    metal::MTLSize::new(tg, 1, 1),
                                );
                                if kernel.starts_with("xcopy") {
                                    &bufs[1]
                                } else {
                                    bufs.last().unwrap()
                                }
                            }
                            Tape::MetaOnly { kernel, bufs, meta, n } => {
                                enc.set_compute_pipeline_state(self.backend.pipeline(kernel));
                                for (kk, bb) in bufs.iter().enumerate() {
                                    enc.set_buffer(kk as u64, Some(bb), 0);
                                }
                                enc.set_bytes(6, 72, meta.as_ptr() as *const _);
                                let tg = 256u64.min((*n).max(1));
                                enc.dispatch_threads(
                                    metal::MTLSize::new((*n).max(1), 1, 1),
                                    metal::MTLSize::new(tg, 1, 1),
                                );
                                bufs.last().unwrap()
                            }
                            Tape::Matmul { .. } => break,
                        };
                        enc.memory_barrier_with_resources(&[written]);
                        i += 1;
                    }
                    enc.end_encoding();
                }
            }
        }
        cb.commit();
        cb.wait_until_completed();
    }

    fn run_meta_only(&self, kernel: &'static str, bufs: &[&Buffer; 6], meta: [i32; 18], n: u64) {
        self.tape.borrow_mut().push(Tape::MetaOnly {
            kernel,
            bufs: bufs.iter().map(|b| (*b).clone()).collect(),
            meta,
            n,
        });
        self.dispatches.set(self.dispatches.get() + 1);
    }

    pub fn poke_i32(&self, id: NodeIndex, vals: &[i32]) {
        let b = &self.vals[&id].buf;
        unsafe {
            std::ptr::copy_nonoverlapping(
                vals.as_ptr() as *const u8,
                b.contents() as *mut u8,
                vals.len() * 4,
            );
        }
    }

    pub fn poke_f32(&self, id: NodeIndex, vals: &[f32]) {
        let b = &self.vals[&id].buf;
        unsafe {
            std::ptr::copy_nonoverlapping(
                vals.as_ptr() as *const u8,
                b.contents() as *mut u8,
                vals.len() * 4,
            );
        }
    }

    pub fn find_zero_scalar_iotas(&self) -> Vec<NodeIndex> {
        self.iota_host
            .iter()
            .filter(|(id, h)| h.len() == 1 && h[0] == 0 && self.dims(**id) == vec![1])
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn read_f32(&mut self, id: NodeIndex) -> Vec<f32> {
        let before = self.tape.borrow().len();
        let v = self.contiguous_val(id);
        assert_eq!(
            self.tape.borrow().len(),
            before,
            "read target must be materialized by the plan, not at read time"
        );
        let n = numel(&self.dims(v)) as usize;
        self.backend.read(&self.vals[&v].buf, n)
    }
}

enum Tape {
    Run1d {
        kernel: &'static str,
        bufs: Vec<Buffer>,
        args: Vec<(u64, Arg)>,
        n: u64,
    },
    MetaOnly {
        kernel: &'static str,
        bufs: Vec<Buffer>,
        meta: [i32; 18],
        n: u64,
    },
    Matmul {
        a: Buffer,
        b: Buffer,
        c: Buffer,
        m: usize,
        k: usize,
        n: usize,
        ap: [i32; 4],
        bp: [i32; 4],
        cp: [i32; 4],
        batch: Vec<i32>,
        nbatch: i64,
    },
}

#[derive(Clone)]
enum Fused {
    Rms { x: NodeIndex, eps: f32 },
    SiluMul { g: NodeIndex, u: NodeIndex, k: f32, c3: f32 },
    Softmax { scores: NodeIndex },
    Chain { primary: NodeIndex, exts: Vec<NodeIndex>, prog: [i32; 25] },
    Attn { q: NodeIndex, k: NodeIndex, v: NodeIndex, bias: NodeIndex, params: [i32; 25] },
}

#[derive(Clone)]
struct MatmulPlan {
    a_base: NodeIndex,
    b_base: NodeIndex,
    a_fix: Option<(NodeIndex, Vec<i64>)>,
    b_fix: Option<(NodeIndex, Vec<i64>)>,
    m: i64,
    k: i64,
    n: i64,
    ap: [i32; 4],
    bp: [i32; 4],
    batch: Vec<i32>,
    nbatch: i64,
}

fn axis_modes(layout: &Layout, ax: usize, rank: usize) -> Vec<(i64, i64)> {
    let mut v = Vec::new();
    if rank <= 1 {
        flat_pairs(&layout.shape, &layout.stride, &mut v);
    } else {
        let m = layout.get(ax);
        flat_pairs(&m.shape, &m.stride, &mut v);
    }
    v.retain(|(sz, _)| *sz != 1);
    v
}

fn group_stride(mut modes_fastest_first: Vec<(i64, i64)>) -> Option<i64> {
    modes_fastest_first.retain(|(sz, _)| *sz != 1);
    if modes_fastest_first.is_empty() {
        return Some(0);
    }
    if modes_fastest_first.iter().all(|(_, d)| *d == 0) {
        return Some(0);
    }
    let s0 = modes_fastest_first[0].1;
    let mut expect = s0;
    for (sz, d) in &modes_fastest_first {
        if *d != expect {
            return None;
        }
        expect = d * sz;
    }
    Some(s0)
}

#[derive(Clone)]
enum Arg {
    Meta([i32; 18]),
    Scalar(u32),
    F32(f32),
    Prog([i32; 25]),
    Spec([i32; 8]),
}
use Arg::{Meta, Scalar, F32, Prog, Spec};
