use crate::cute_probe::probe;
use crate::metal_backend::CuteMetal;
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

    fn plan_matmuls(&mut self) {
        let ids: Vec<NodeIndex> = self.g.node_indices().collect();
        for rs in ids {
            let LogicalOp::ReduceSum { axis_from_end } = self.g[rs].op else {
                continue;
            };
            let mul = self.operands(rs)[0];
            if !matches!(self.g[mul].op, LogicalOp::Mul) {
                continue;
            }
            if self.g.neighbors_directed(mul, Direction::Outgoing).count() != 1 {
                continue;
            }
            let ops = self.operands(mul);
            let dims = self.dims(mul);
            let rank = dims.len();
            let ax = rank - 1 - axis_from_end;
            let Some((a_base, al)) = self.resolve_dry(ops[0]) else {
                continue;
            };
            let Some((b_base, bl)) = self.resolve_dry(ops[1]) else {
                continue;
            };
            let varies = |l: &Layout, axis: usize| {
                axis_modes(l, axis, rank).iter().any(|(_, d)| *d != 0)
            };
            if !varies(&al, ax) || !varies(&bl, ax) {
                continue;
            }
            let (mut m_axes, mut n_axes) = (Vec::new(), Vec::new());
            let mut ok = true;
            for axis in 0..rank {
                if axis == ax {
                    continue;
                }
                match (varies(&al, axis), varies(&bl, axis)) {
                    (true, true) => {
                        ok = false;
                        break;
                    }
                    (_, true) => n_axes.push(axis),
                    _ => m_axes.push(axis),
                }
            }
            if !ok || m_axes.iter().any(|ma| n_axes.iter().any(|na| ma > na)) {
                continue;
            }
            let collect = |l: &Layout, axes: &[usize]| -> Vec<(i64, i64)> {
                let mut v = Vec::new();
                for axis in axes.iter().rev() {
                    v.extend(axis_modes(l, *axis, rank));
                }
                v
            };
            let Some(ars) = group_stride(collect(&al, &m_axes)) else {
                continue;
            };
            let Some(acs) = group_stride(collect(&al, &[ax])) else {
                continue;
            };
            let Some(brs) = group_stride(collect(&bl, &[ax])) else {
                continue;
            };
            let Some(bcs) = group_stride(collect(&bl, &n_axes)) else {
                continue;
            };
            let m: i64 = m_axes.iter().map(|x| dims[*x]).product();
            let n: i64 = n_axes.iter().map(|x| dims[*x]).product();
            let k = dims[ax];
            let fits = |v: i64| i32::try_from(v).is_ok();
            if ![ars, acs, brs, bcs, m, n, k].iter().all(|v| fits(*v)) {
                continue;
            }
            self.mm_skip.insert(mul);
            self.mm_plan.insert(
                rs,
                MatmulPlan {
                    a_base,
                    b_base,
                    m,
                    k,
                    n,
                    ap: [0, ars as i32, acs as i32, 0],
                    bp: [0, brs as i32, bcs as i32, 0],
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

    fn materialize_value(&mut self, id: NodeIndex, base: NodeIndex, acc: Layout) -> NodeIndex {
        if self.vals.contains_key(&id) {
            return id;
        }
        let dims = self.dims(id);
        let n = numel(&dims);
        let dtype = self.g[id].dtype;
        let meta = meta_of(&acc, dims.len()).expect("materialize meta");
        let kernel = match dtype {
            DType::F32 => "copy_strided",
            DType::Int => "copy_strided_i",
            other => panic!("materialize dtype {other:?}"),
        };
        let bytes = n as usize * 4;
        let out = self.alloc(bytes);
        let src = &self.vals[&base].buf;
        self.run1d(kernel, &[src, &out], &[(2, Scalar(n as u32)), (3, Meta(meta))], n as u64);
        self.materializes.set(self.materializes.get() + 1);
        self.vals.insert(id, Val { buf: out, dims, dtype });
        self.access.insert(id, (id, identity_layout(&self.dims(id))));
        id
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
                Tape::Run1d { kernel, n, .. } => {
                    let n = *n as usize;
                    match *kernel {
                        "add_strided" | "mul_strided" => 3 * n * 4,
                        "copy_strided" | "copy_strided_i" | "recip_o" | "sqrt_o" | "exp2_o"
                        | "cast_i2f" => 2 * n * 4,
                        "cast_b2f" => n + n * 4,
                        "less_than_i" => 2 * n * 4 + n,
                        "reduce_sum_axis" | "reduce_max_axis" => n * 4,
                        other => panic!("traffic for {other}"),
                    }
                }
                Tape::MetaOnly { n, .. } => {
                    let n = *n as usize;
                    n * 4 * 2 + n * 4 * 2
                }
                Tape::Matmul { m, k, n, .. } => (m * k + k * n + m * n) * 4,
            })
            .sum()
    }

    pub fn replay(&self) {
        let cb = self.backend.owned_command_buffer();
        for t in self.tape.borrow().iter() {
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
                Tape::Matmul { a, b, c, m, k, n, ap, bp, cp } => {
                    self.backend.matmul_into(&cb, a, b, c, *m, *k, *n, *ap, *bp, *cp);
                }
            }
        }
        cb.commit();
        cb.wait_until_completed();
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

    pub fn run(&mut self, inputs: &FxHashMap<NodeIndex, luminal_metal::HostBuffer>) {
        self.plan_matmuls();
        let order = toposort(&self.g, None).expect("logical graph is a DAG");
        for id in order {
            if self.mm_skip.contains(&id) {
                continue;
            }
            if let Some(plan) = self.mm_plan.get(&id).cloned() {
                let dims = self.dims(id);
                let n_out = numel(&dims);
                assert_eq!(n_out, plan.m * plan.n, "matmul out extents");
                let out = self.alloc(n_out as usize * 4);
                self.tape.borrow_mut().push(Tape::Matmul {
                    a: self.vals[&plan.a_base].buf.clone(),
                    b: self.vals[&plan.b_base].buf.clone(),
                    c: out.clone(),
                    m: plan.m as usize,
                    k: plan.k as usize,
                    n: plan.n as usize,
                    ap: plan.ap,
                    bp: plan.bp,
                    cp: [0, plan.n as i32, 1, 0],
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
                    self.vals.insert(id, Val { buf, dims, dtype: DType::Int });
                }
                LogicalOp::Cast(dt) | LogicalOp::RoundingCast(dt) => {
                    assert_eq!(*dt, DType::F32, "only casts to F32 appear in this model");
                    let src = self.operands(id)[0];
                    let sv = self.contiguous_val(src);
                    let kernel = match self.vals[&sv].dtype {
                        DType::Int => "cast_i2f",
                        DType::Bool => "cast_b2f",
                        other => panic!("cast from {other:?}"),
                    };
                    let out = self.alloc(n as usize * 4);
                    let sb = &self.vals[&sv].buf;
                    self.run1d(kernel, &[sb, &out], &[(2, Scalar(n as u32))], n as u64);
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
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
                    let av = self.contiguous_val(ops[0]);
                    let bv = self.contiguous_val(ops[1]);
                    let out = self.alloc(n as usize);
                    let (abuf, bbuf) = (&self.vals[&av].buf, &self.vals[&bv].buf);
                    self.run1d("less_than_i", &[abuf, bbuf, &out], &[(3, Scalar(n as u32))], n as u64);
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::Bool });
                }
                LogicalOp::ReduceSum { axis_from_end } | LogicalOp::ReduceMax { axis_from_end } => {
                    let kernel = if matches!(self.g[id].op, LogicalOp::ReduceSum { .. }) {
                        "reduce_sum_axis"
                    } else {
                        "reduce_max_axis"
                    };
                    let src = self.operands(id)[0];
                    let sv = self.contiguous_val(src);
                    let sdims = self.dims(sv);
                    let axis = sdims.len() - 1 - axis_from_end;
                    let outer: i64 = sdims[..axis].iter().product();
                    let axlen = sdims[axis];
                    let inner: i64 = sdims[axis + 1..].iter().product();
                    assert_eq!(outer * inner, n, "reduce extents");
                    let out = self.alloc(n as usize * 4);
                    let sb = &self.vals[&sv].buf;
                    self.run1d(
                        kernel,
                        &[sb, &out],
                        &[
                            (2, Scalar(outer as u32)),
                            (3, Scalar(axlen as u32)),
                            (4, Scalar(inner as u32)),
                        ],
                        n as u64,
                    );
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                }
                LogicalOp::Gather => {
                    let ops = self.operands(id);
                    let dv = self.contiguous_val(ops[0]);
                    let ddims = self.dims(dv);
                    let rank = ddims.len();
                    assert!(rank <= 4, "gather rank cap");
                    assert_eq!(ops.len(), rank + 1, "gather operands = data + rank coords");
                    let coords: Vec<NodeIndex> =
                        ops[1..].iter().map(|c| self.contiguous_val(*c)).collect();
                    let strides = rm_strides(&ddims);
                    let mut meta = [0i32; 18];
                    meta[0] = rank as i32;
                    meta[1] = n as i32;
                    for (k, s) in strides.iter().enumerate() {
                        meta[2 + k] = *s as i32;
                    }
                    let out = self.alloc(n as usize * 4);
                    let dbuf = &self.vals[&dv].buf;
                    let cbufs: Vec<&Buffer> = (0..4)
                        .map(|k| &self.vals[coords.get(k).unwrap_or(&coords[0])].buf)
                        .collect();
                    let bufs = [dbuf, cbufs[0], cbufs[1], cbufs[2], cbufs[3], &out];
                    self.run_meta_only("gather4", &bufs, meta, n as u64);
                    self.vals.insert(id, Val { buf: out, dims, dtype: DType::F32 });
                }
                LogicalOp::Scatter => {
                    let ops = self.operands(id);
                    let (ibase, iacc) = self.resolve(ops[0]);
                    let idims = self.dims(ops[0]);
                    assert_eq!(idims, dims, "scatter init matches out");
                    let imeta = meta_of(&iacc, idims.len()).expect("scatter init meta");
                    let out = self.alloc(n as usize * 4);
                    let ibuf = &self.vals[&ibase].buf;
                    self.run1d("copy_strided", &[ibuf, &out], &[(2, Scalar(n as u32)), (3, Meta(imeta))], n as u64);
                    let src = *ops.last().unwrap();
                    let sv = self.contiguous_val(src);
                    let sdims = self.dims(sv);
                    let src_n = numel(&sdims);
                    let rank = dims.len();
                    assert!(rank <= 4, "scatter rank cap");
                    assert_eq!(ops.len(), rank + 2, "scatter operands = init + coords + src");
                    let coords: Vec<NodeIndex> = ops[1..ops.len() - 1]
                        .iter()
                        .map(|c| self.contiguous_val(*c))
                        .collect();
                    let strides = rm_strides(&dims);
                    let mut meta = [0i32; 18];
                    meta[0] = rank as i32;
                    meta[1] = src_n as i32;
                    for (k, s) in strides.iter().enumerate() {
                        meta[2 + k] = *s as i32;
                    }
                    let sbuf = &self.vals[&sv].buf;
                    let cbufs: Vec<&Buffer> = (0..4)
                        .map(|k| &self.vals[coords.get(k).unwrap_or(&coords[0])].buf)
                        .collect();
                    let bufs = [sbuf, cbufs[0], cbufs[1], cbufs[2], cbufs[3], &out];
                    self.run_meta_only("scatter4", &bufs, meta, src_n as u64);
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

    fn run_meta_only(&self, kernel: &'static str, bufs: &[&Buffer; 6], meta: [i32; 18], n: u64) {
        self.tape.borrow_mut().push(Tape::MetaOnly {
            kernel,
            bufs: bufs.iter().map(|b| (*b).clone()).collect(),
            meta,
            n,
        });
        self.dispatches.set(self.dispatches.get() + 1);
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
    },
}

#[derive(Clone)]
struct MatmulPlan {
    a_base: NodeIndex,
    b_base: NodeIndex,
    m: i64,
    k: i64,
    n: i64,
    ap: [i32; 4],
    bp: [i32; 4],
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
}
use Arg::{Meta, Scalar};
