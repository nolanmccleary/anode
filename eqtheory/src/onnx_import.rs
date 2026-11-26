use crate::ir::{layout_of, rm_strides, GraphIr, Nl, Read, Tensor};
use crate::ir::Dt;
use crate::onnx_proto::{ModelProto, NodeProto, TensorProto};
use prost::Message;
use std::collections::HashMap;

const LOG2E: f32 = std::f32::consts::LOG2_E;

pub struct ParsedModel {
    pub g: GraphIr,
    pub weights: HashMap<String, Vec<f32>>,
    pub runtime_inputs: Vec<(String, Vec<i64>, Dt)>,
    pub outputs: Vec<(String, usize)>,
    pub input_nodes: HashMap<String, usize>,
    pub logits: usize,
    pub cache_ins: Vec<usize>,
    pub cache_outs: Vec<usize>,
    pub probes: Vec<(String, usize)>,
}

#[derive(Clone, Copy)]
pub struct CacheBind {
    pub cap: i64,
    pub ctx: i64,
}

#[derive(Clone)]
struct Handle {
    node: usize,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

fn contiguous(h: &Handle) -> bool {
    h.strides == rm_strides(&h.shape)
}

fn ten_f32(t: &TensorProto) -> Vec<f32> {
    if !t.float_data.is_empty() {
        return t.float_data.clone();
    }
    let raw = t.raw_data();
    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn ten_i64(t: &TensorProto) -> Vec<i64> {
    if !t.int64_data.is_empty() {
        return t.int64_data.clone();
    }
    let raw = t.raw_data();
    raw.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

struct Lowerer<'a> {
    g: GraphIr,
    env: HashMap<String, Handle>,
    inits: HashMap<String, &'a TensorProto>,
    weights: HashMap<String, Vec<f32>>,
    runtime_inputs: Vec<(String, Vec<i64>, Dt)>,
    scalar: HashMap<String, f32>,
    ishape: HashMap<String, Vec<i64>>,
    deferred_sqrt: HashMap<String, Handle>,
    cache: Option<CacheBind>,
    scatter_node: Option<usize>,
    gather_node: Option<usize>,
    cache_nodes: HashMap<String, usize>,
    cache_ins: Vec<usize>,
    cache_outs: Vec<usize>,
    probes: Vec<(String, usize)>,
    resid_count: usize,
}

impl<'a> Lowerer<'a> {
    fn tensor(&self, h: &Handle) -> Tensor {
        Tensor { id: h.node }
    }

    fn contig(&mut self, h: &Handle) -> Handle {
        if contiguous(h) {
            return h.clone();
        }
        let r = Read { src: h.node, layout: layout_of(&h.shape, &h.strides), base: 0, nls: Vec::new() };
        let t = self.g.movn(h.shape.clone(), r, None, 1, true);
        Handle { node: t.id, shape: h.shape.clone(), strides: rm_strides(&h.shape) }
    }

    fn fresh(&self, t: Tensor) -> Handle {
        let dims = self.g.nodes[t.id].dims.clone();
        Handle { node: t.id, strides: rm_strides(&dims), shape: dims }
    }

    fn read(&self, h: &Handle) -> Read {
        Read { src: h.node, layout: layout_of(&h.shape, &h.strides), base: 0, nls: Vec::new() }
    }

    fn get(&self, name: &str) -> Handle {
        self.env.get(name).unwrap_or_else(|| panic!("onnx: undefined tensor {name}")).clone()
    }

    fn matmul(&mut self, a: &Handle, b: &Handle) -> Handle {
        let ra = a.shape.len();
        let rb = b.shape.len();
        let (m, ka) = (a.shape[ra - 2], a.shape[ra - 1]);
        let (kb, n) = (b.shape[rb - 2], b.shape[rb - 1]);
        assert_eq!(ka, kb, "onnx matmul k");
        let abatch = &a.shape[..ra - 2];
        let bbatch = &b.shape[..rb - 2];
        let nb = abatch.len().max(bbatch.len());
        let mut out_batch = vec![0i64; nb];
        let mut a_bs = vec![0i64; nb];
        let mut b_bs = vec![0i64; nb];
        for i in 0..nb {
            let ai = (abatch.len() as isize - nb as isize + i as isize) as isize;
            let bi = (bbatch.len() as isize - nb as isize + i as isize) as isize;
            let (adim, astr) = if ai >= 0 { (abatch[ai as usize], a.strides[ai as usize]) } else { (1, 0) };
            let (bdim, bstr) = if bi >= 0 { (bbatch[bi as usize], b.strides[bi as usize]) } else { (1, 0) };
            let od = adim.max(bdim);
            out_batch[i] = od;
            a_bs[i] = if adim == 1 { 0 } else { astr };
            b_bs[i] = if bdim == 1 { 0 } else { bstr };
        }
        let mut oshape = out_batch.clone();
        oshape.push(m);
        oshape.push(n);
        let mut a_str = a_bs.clone();
        a_str.push(a.strides[ra - 2]);
        a_str.push(0);
        a_str.push(a.strides[ra - 1]);
        let mut b_str = b_bs.clone();
        b_str.push(0);
        b_str.push(b.strides[rb - 1]);
        b_str.push(b.strides[rb - 2]);
        let mut rshape = oshape.clone();
        rshape.push(ka);
        let ra_read = Read { src: a.node, layout: layout_of(&rshape, &a_str), base: 0, nls: Vec::new() };
        let rb_read = Read { src: b.node, layout: layout_of(&rshape, &b_str), base: 0, nls: Vec::new() };
        let t = self.g.movn(oshape.clone(), ra_read, Some(rb_read), ka, true);
        self.fresh(t)
    }

    fn binary(&mut self, a: &Handle, b: &Handle, pair: bool) -> Handle {
        let nd = a.shape.len().max(b.shape.len());
        let mut oshape = vec![0i64; nd];
        let mut a_str = vec![0i64; nd];
        let mut b_str = vec![0i64; nd];
        for i in 0..nd {
            let ai = a.shape.len() as isize - nd as isize + i as isize;
            let bi = b.shape.len() as isize - nd as isize + i as isize;
            let (ad, ast) = if ai >= 0 { (a.shape[ai as usize], a.strides[ai as usize]) } else { (1, 0) };
            let (bd, bst) = if bi >= 0 { (b.shape[bi as usize], b.strides[bi as usize]) } else { (1, 0) };
            oshape[i] = ad.max(bd);
            a_str[i] = if ad == 1 { 0 } else { ast };
            b_str[i] = if bd == 1 { 0 } else { bst };
        }
        if pair {
            let mut psh = oshape.clone();
            psh.push(1);
            let mut pa = a_str.clone();
            pa.push(0);
            let mut pb = b_str.clone();
            pb.push(0);
            let ra = Read { src: a.node, layout: layout_of(&psh, &pa), base: 0, nls: Vec::new() };
            let rb = Read { src: b.node, layout: layout_of(&psh, &pb), base: 0, nls: Vec::new() };
            let t = self.g.movn(oshape.clone(), ra, Some(rb), 1, true);
            return self.fresh(t);
        }
        let mut sshape = oshape.clone();
        sshape.push(2);
        a_str.push(0);
        b_str.push(0);
        let ra = Read { src: a.node, layout: layout_of(&sshape, &a_str), base: 0, nls: Vec::new() };
        let rb = Read { src: b.node, layout: layout_of(&sshape, &b_str), base: 0, nls: Vec::new() };
        let t = self.g.movn(oshape.clone(), ra, Some(rb), 2, false);
        self.fresh(t)
    }

    fn lower(&mut self, node: &NodeProto) {
        let op = node.op_type();
        let ins: Vec<String> = node.input.clone();
        let out = node.output[0].clone();
        match op {
            "Gather" => {
                let table = self.get(&ins[0]);
                let idx = self.get(&ins[1]);
                let rows = idx.shape[0];
                let t = self.g.gather_rows(self.tensor(&table), self.tensor(&idx), rows);
                self.env.insert(out, self.fresh(t));
            }
            "MatMul" => {
                let a = self.get(&ins[0]);
                let b = self.get(&ins[1]);
                let h = self.matmul(&a, &b);
                if let Some(rest) = ins[1].strip_prefix("model.layers.") {
                    if let Some(ln) = rest.strip_suffix(".self_attn.q_proj.weight") {
                        self.probes.push((format!("q.{ln}"), h.node));
                    }
                }
                self.env.insert(out, h);
            }
            "Mul" => {
                if let Some(c) = self.scalar.get(&ins[1]).or_else(|| self.scalar.get(&ins[0])) {
                    let c = *c;
                    let t = if self.scalar.contains_key(&ins[1]) { self.get(&ins[0]) } else { self.get(&ins[1]) };
                    let t = self.contig(&t);
                    let r = self.g.scale(self.tensor(&t), c);
                    self.env.insert(out, self.fresh(r));
                } else {
                    let a = self.get(&ins[0]);
                    let b = self.get(&ins[1]);
                    let h = self.binary(&a, &b, true);
                    self.env.insert(out, h);
                }
            }
            "Add" => {
                if let Some(c) = self.scalar.get(&ins[1]).or_else(|| self.scalar.get(&ins[0])) {
                    let c = *c;
                    let t = if self.scalar.contains_key(&ins[1]) { self.get(&ins[0]) } else { self.get(&ins[1]) };
                    let t = self.contig(&t);
                    let lit = self.g.lit_f32(vec![c], vec![1]);
                    let r = self.g.add_bcast(self.tensor(&t), lit);
                    self.env.insert(out, self.fresh(r));
                } else {
                    let a = self.get(&ins[0]);
                    let b = self.get(&ins[1]);
                    let h = self.binary(&a, &b, false);
                    if h.shape.len() == 2 {
                        let n = self.resid_count;
                        let label = if n % 2 == 0 { format!("resid_attn.{}", n / 2) } else { format!("resid_ff.{}", n / 2) };
                        self.probes.push((label, h.node));
                        self.resid_count += 1;
                    }
                    self.env.insert(out, h);
                }
            }
            "Sqrt" => {
                let x = self.get(&ins[0]);
                self.deferred_sqrt.insert(out, x);
            }
            "Div" => {
                let a = self.get(&ins[0]);
                if let Some(y) = self.deferred_sqrt.get(&ins[1]).cloned() {
                    let yc = self.contig(&y);
                    let r = self.g.pointwise(Nl::Rsqrt, self.tensor(&yc));
                    let h = self.g.mul_bcast_row2(self.tensor(&a), r);
                    self.env.insert(out, self.fresh(h));
                } else {
                    let b = self.get(&ins[1]);
                    let bc = self.contig(&b);
                    let rb = self.g.pointwise(Nl::Recip, self.tensor(&bc));
                    let h = self.binary(&a, &self.fresh(rb), true);
                    self.env.insert(out, h);
                }
            }
            "ReduceMean" => {
                let x = self.contig(&self.get(&ins[0]));
                let d = *x.shape.last().unwrap();
                let s = self.g.sum_last(self.tensor(&x));
                let ms = self.g.scale(s, 1.0 / d as f32);
                self.env.insert(out, self.fresh(ms));
            }
            "Sigmoid" => {
                let x = self.contig(&self.get(&ins[0]));
                let neg = self.g.scale(self.tensor(&x), -LOG2E);
                let e = self.g.pointwise(Nl::Exp2, neg);
                let one = self.g.lit_f32(vec![1.0], vec![1]);
                let denom = self.g.add_bcast(e, one);
                let sig = self.g.pointwise(Nl::Recip, denom);
                self.env.insert(out, self.fresh(sig));
            }
            "Softmax" => {
                let x = self.contig(&self.get(&ins[0]));
                let scaled = self.g.scale(self.tensor(&x), LOG2E);
                let rowmax = self.g.max_last(scaled);
                let negmax = self.g.scale(rowmax, -1.0);
                let shifted = self.g.add_bcast_row2(scaled, negmax);
                let ex = self.g.pointwise(Nl::Exp2, shifted);
                let denom = self.g.sum_last(ex);
                let rd = self.g.pointwise(Nl::Recip, denom);
                let probs = self.g.mul_bcast_row2(ex, rd);
                self.env.insert(out, self.fresh(probs));
            }
            "Reshape" => {
                let x = self.get(&ins[0]);
                let target = ten_i64(self.inits[&ins[1]]);
                let total: i64 = x.shape.iter().product();
                let mut shape: Vec<i64> = target
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| if t == 0 { x.shape[i] } else { t })
                    .collect();
                let known: i64 = shape.iter().filter(|&&s| s != -1).product();
                for s in shape.iter_mut() {
                    if *s == -1 {
                        *s = total / known;
                    }
                }
                let xc = if contiguous(&x) { x } else { self.contig(&x) };
                self.env.insert(out, Handle { node: xc.node, strides: rm_strides(&shape), shape });
            }
            "Transpose" => {
                let x = self.get(&ins[0]);
                let perm: Vec<usize> = node
                    .attribute
                    .iter()
                    .find(|a| a.name() == "perm")
                    .map(|a| a.ints.iter().map(|&i| i as usize).collect())
                    .unwrap_or_else(|| (0..x.shape.len()).rev().collect());
                let shape = perm.iter().map(|&p| x.shape[p]).collect();
                let strides = perm.iter().map(|&p| x.strides[p]).collect();
                self.env.insert(out, Handle { node: x.node, shape, strides });
            }
            "Unsqueeze" => {
                let x = self.get(&ins[0]);
                let axes: Vec<i64> = ten_i64(self.inits[&ins[1]]);
                let nd = x.shape.len() + axes.len();
                let axset: Vec<usize> =
                    axes.iter().map(|&a| if a < 0 { (a + nd as i64) as usize } else { a as usize }).collect();
                let mut shape = Vec::new();
                let mut strides = Vec::new();
                let mut src = 0usize;
                for i in 0..nd {
                    if axset.contains(&i) {
                        shape.push(1);
                        strides.push(0);
                    } else {
                        shape.push(x.shape[src]);
                        strides.push(x.strides[src]);
                        src += 1;
                    }
                }
                self.env.insert(out, Handle { node: x.node, shape, strides });
            }
            "Concat" => {
                let axis = node.attribute.iter().find(|a| a.name() == "axis").map(|a| a.i()).unwrap_or(0);
                assert_eq!(axis, 0, "onnx concat axis");
                match self.cache {
                    None => {
                        let h = self.get(&ins[1]);
                        self.env.insert(out, h);
                    }
                    Some(cb) => {
                        let newk = self.get(&ins[1]);
                        let kvd = *newk.shape.last().unwrap();
                        let cin = *self.cache_nodes.entry(ins[0].clone()).or_insert_with(|| {
                            let c = self.g.input(&ins[0], vec![cb.cap, kvd], Dt::F32);
                            c.id
                        });
                        self.cache_ins.push(cin);
                        let si = Tensor { id: self.scatter_node.unwrap() };
                        let gi = Tensor { id: self.gather_node.unwrap() };
                        let kc = self.g.scatter_rows(Tensor { id: cin }, si, self.tensor(&newk));
                        let keys = self.g.gather_rows(kc, gi, cb.ctx);
                        self.cache_outs.push(kc.id);
                        let kh = self.fresh(keys);
                        if let Some(rest) = ins[0].strip_prefix("past_k_") {
                            self.probes.push((format!("keys.{rest}"), kh.node));
                        }
                        self.env.insert(out, kh);
                    }
                }
            }
            other => panic!("onnx: unsupported op {other}"),
        }
    }
}

pub fn parse_bytes(bytes: &[u8], seq: i64) -> ParsedModel {
    let model = ModelProto::decode(bytes).expect("decode onnx");
    parse(&model, seq, None)
}

pub fn parse(model: &ModelProto, seq: i64, cache: Option<CacheBind>) -> ParsedModel {
    let graph = model.graph.as_ref().expect("graph");
    let weight_names: Vec<&str> = graph
        .initializer
        .iter()
        .filter(|t| t.data_type() == 1 && t.dims.iter().product::<i64>() > 1)
        .map(|t| t.name())
        .collect();
    let weight_set: std::collections::HashSet<String> = weight_names.iter().map(|s| s.to_string()).collect();

    let mut low = Lowerer {
        g: GraphIr::new(),
        env: HashMap::new(),
        inits: graph.initializer.iter().map(|t| (t.name().to_string(), t)).collect(),
        weights: HashMap::new(),
        runtime_inputs: Vec::new(),
        scalar: HashMap::new(),
        ishape: HashMap::new(),
        deferred_sqrt: HashMap::new(),
        cache,
        scatter_node: None,
        gather_node: None,
        cache_nodes: HashMap::new(),
        cache_ins: Vec::new(),
        cache_outs: Vec::new(),
        probes: Vec::new(),
        resid_count: 0,
    };
    let mut input_nodes: HashMap<String, usize> = HashMap::new();

    for t in &graph.initializer {
        if weight_set.contains(t.name()) {
            let dims: Vec<i64> = t.dims.clone();
            let tn = low.g.input(t.name(), dims.clone(), Dt::F32);
            low.weights.insert(t.name().to_string(), ten_f32(t));
            input_nodes.insert(t.name().to_string(), tn.id);
            low.env.insert(t.name().to_string(), low.fresh(tn));
        } else if t.data_type() == 1 && t.dims.iter().product::<i64>() <= 1 {
            low.scalar.insert(t.name().to_string(), ten_f32(t)[0]);
        }
    }

    if let Some(cb) = cache {
        let si = low.g.input("scatter_idx", vec![seq], Dt::I32);
        input_nodes.insert("scatter_idx".into(), si.id);
        low.runtime_inputs.push(("scatter_idx".into(), vec![seq], Dt::I32));
        low.scatter_node = Some(si.id);
        let gi = low.g.lit_i32((0..cb.ctx as i32).collect(), vec![cb.ctx]);
        low.gather_node = Some(gi.id);
    }

    let total = cache.map(|c| c.ctx).unwrap_or(seq);
    for vi in &graph.input {
        let name = vi.name().to_string();
        if low.env.contains_key(&name) {
            continue;
        }
        if name.starts_with("past_k_") || name.starts_with("past_v_") {
            continue;
        }
        let shape = resolve_input_shape(vi, seq, total);
        low.ishape.insert(name.clone(), shape.clone());
        if shape.iter().product::<i64>() == 0 {
            continue;
        }
        let dt = if name == "input_ids" { Dt::I32 } else { Dt::F32 };
        let tn = low.g.input(&name, shape.clone(), dt);
        input_nodes.insert(name.clone(), tn.id);
        low.runtime_inputs.push((name.clone(), shape.clone(), dt));
        low.env.insert(name, low.fresh(tn));
    }

    for node in &graph.node {
        low.lower(node);
    }
    for (name, id) in &low.cache_nodes {
        input_nodes.insert(name.clone(), *id);
    }

    let outputs: Vec<(String, usize)> = graph
        .output
        .iter()
        .filter_map(|o| low.env.get(o.name()).map(|h| (o.name().to_string(), h.node)))
        .collect();
    let logits = outputs.iter().find(|(n, _)| n == "logits").map(|(_, id)| *id).expect("logits output");

    ParsedModel {
        g: low.g,
        weights: low.weights,
        runtime_inputs: low.runtime_inputs,
        outputs,
        input_nodes,
        logits,
        cache_ins: low.cache_ins,
        cache_outs: low.cache_outs,
        probes: low.probes,
    }
}

fn resolve_input_shape(vi: &crate::onnx_proto::ValueInfoProto, seq: i64, total: i64) -> Vec<i64> {
    let tt = vi.r#type.as_ref().and_then(|t| t.value.as_ref());
    use crate::onnx_proto::type_proto::Value;
    let Some(Value::TensorType(tensor)) = tt else {
        return vec![];
    };
    let shape = tensor.shape.as_ref().expect("input shape");
    shape
        .dim
        .iter()
        .map(|d| {
            use crate::onnx_proto::tensor_shape_proto::dimension::Value as DV;
            match &d.value {
                Some(DV::DimValue(v)) => *v,
                Some(DV::DimParam(p)) => {
                    if p == "seq" {
                        seq
                    } else if p == "total" {
                        total
                    } else {
                        0
                    }
                }
                None => 0,
            }
        })
        .collect()
}
