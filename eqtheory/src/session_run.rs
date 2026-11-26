use crate::native_exec::NativeExec;
use crate::onnx_import::{parse, CacheBind, ParsedModel};
use crate::onnx_proto::ModelProto;
use prost::Message;
use std::collections::HashMap;
use std::time::Instant;

pub struct Workload {
    pub prompt: Vec<i32>,
    pub gen_tokens: Vec<i32>,
    pub ctx: i64,
}

pub struct SessionMetrics {
    pub ttft_ms: f64,
    pub decode_tok_s: f64,
    pub e2e_ms: f64,
    pub prefill_dispatches: usize,
    pub decode_dispatches: usize,
    pub last_logits: Vec<f32>,
    pub first_logits: Vec<f32>,
    pub final_logits: Vec<f32>,
}

fn mask_row(pos: i64, ctx: i64) -> Vec<f32> {
    (0..ctx).map(|c| if c > pos { -1e10 } else { 0.0 }).collect()
}

fn causal(seq: i64, ctx: i64) -> Vec<f32> {
    let mut m = vec![0f32; (seq * ctx) as usize];
    for q in 0..seq {
        for c in 0..ctx {
            if c > q {
                m[(q * ctx + c) as usize] = -1e10;
            }
        }
    }
    m
}

fn weights_as_inputs(pm: &ParsedModel) -> HashMap<String, Vec<f32>> {
    pm.weights.clone()
}

fn add_concat(pin: &mut HashMap<String, Vec<f32>>, recipes: &[crate::cogroup::ConcatRecipe]) {
    for r in recipes {
        let mut v = vec![0f32; (r.k * r.n_total) as usize];
        let mut off = 0i64;
        for (ci, name) in r.comps.iter().enumerate() {
            let w = &pin[name];
            let n_x = r.n_each[ci];
            let (bn, bk) = (r.bn[ci], r.bk[ci]);
            for ki in 0..r.k {
                for j in 0..n_x {
                    v[(ki * r.n_total + off + j) as usize] = w[(j * bn + ki * bk) as usize];
                }
            }
            off += n_x;
        }
        pin.insert(r.out_name.clone(), v);
    }
}

pub fn run_session(onnx_bytes: &[u8], w: &Workload) -> SessionMetrics {
    let model = ModelProto::decode(onnx_bytes).expect("decode onnx");
    let p = w.prompt.len() as i64;
    let g = w.gen_tokens.len();
    let vocab = {
        let out = model.graph.as_ref().unwrap().output.iter().find(|o| o.name() == "logits").unwrap();
        resolve_vocab(out)
    };

    let mut pre = parse(&model, p, Some(CacheBind { cap: w.ctx, ctx: w.ctx }));
    if std::env::var("MLXFULL").is_ok() {
        return run_session_mlx(w, &pre.weights, vocab);
    }
    let (pre_recipes, pre_dead) = if std::env::var("COGROUP").is_ok() {
        crate::cogroup::coarsen_cogroup(&mut pre.g)
    } else {
        (Vec::new(), Default::default())
    };
    let mut pin = weights_as_inputs(&pre);
    add_concat(&mut pin, &pre_recipes);
    pin.insert("input_ids".into(), w.prompt.iter().map(|t| *t as f32).collect());
    pin.insert("scatter_idx".into(), (0..p).map(|i| i as f32).collect());
    pin.insert("attn_mask".into(), causal(p, w.ctx));
    for id in &pre.cache_ins {
        let name = pre.input_nodes.iter().find(|(_, v)| **v == *id).map(|(n, _)| n.clone()).unwrap();
        pin.insert(name, vec![0.0; (w.ctx * kvd_of(&pre, *id)) as usize]);
    }
    if std::env::var("CPU_REF").is_ok() {
        let refm = crate::cpu_ref::eval(&pre.g, &pin);
        let all = refm.f[pre.logits].as_ref().unwrap();
        let last = all[((p - 1) * vocab) as usize..].to_vec();
        return SessionMetrics {
            ttft_ms: 0.0, decode_tok_s: 0.0, e2e_ms: 0.0,
            prefill_dispatches: 0, decode_dispatches: 0,
            last_logits: last.clone(), first_logits: last.clone(), final_logits: last,
        };
    }
    let mut preex = NativeExec::new();
    preex.dead = pre_dead;
    preex.build(&pre.g, &pin);
    let t0 = Instant::now();
    preex.replay();
    let ttft = t0.elapsed().as_secs_f64() * 1e3;
    if std::env::var("GRAPHDUMP").is_ok() {
        for line in crate::compose::dump(&pre.g) {
            eprintln!("{line}");
        }
    }
    if std::env::var("REFCHK").is_ok() {
        let refm = crate::cpu_ref::eval(&pre.g, &pin);
        match preex.first_divergence(&pre.g, &refm) {
            Some((id, rel)) => {
                eprintln!("REFCHK prefill: FIRST divergence node {id} rel={rel:.2e}: {}", describe(&pre.g, id));
                let show = |nid: usize| {
                    let nv = preex.read_f32(&pre.g, nid);
                    let rv = refm.f[nid].as_ref().unwrap();
                    eprintln!("  node {nid} native[0..6]={:?}", &nv[..6.min(nv.len())]);
                    eprintln!("  node {nid} cpuref[0..6]={:?}", &rv[..6.min(rv.len())]);
                };
                show(id);
                if let crate::ir::NodeKind::Move { a, b, .. } = &pre.g.nodes[id].kind {
                    eprintln!("  a.src {}: {}", a.src, describe(&pre.g, a.src));
                    show(a.src);
                    if let Some(b) = b {
                        if b.src != a.src {
                            show(b.src);
                        }
                    }
                }
            }
            None => eprintln!("REFCHK prefill: native matches cpu_ref everywhere"),
        }
    }
    if std::env::var("NANSCAN").is_ok() {
        match preex.first_nan(&pre.g) {
            Some((id, k)) => {
                eprintln!("NANSCAN prefill: FIRST NaN node {id} elem {k}: {}", describe(&pre.g, id));
                let stat = |nid: usize| {
                    let v = preex.read_f32(&pre.g, nid);
                    let maxabs = v.iter().filter(|x| x.is_finite()).fold(0f32, |m, x| m.max(x.abs()));
                    let minv = v.iter().filter(|x| x.is_finite()).fold(f32::INFINITY, |m, &x| m.min(x));
                    let infc = v.iter().filter(|x| x.is_infinite()).count();
                    eprintln!("  node {nid}: maxabs={maxabs:.3e} min={minv:.3e} inf={infc} :: {}", describe(&pre.g, nid));
                };
                let mut cur = id;
                for _ in 0..8 {
                    stat(cur);
                    match &pre.g.nodes[cur].kind {
                        crate::ir::NodeKind::Move { a, b, .. } => {
                            if let Some(b) = b {
                                stat(b.src);
                            }
                            cur = a.src;
                        }
                        crate::ir::NodeKind::Pointwise { read, .. } => cur = read.src,
                        _ => break,
                    }
                }
            }
            None => eprintln!("NANSCAN prefill: no NaN in any materialized buffer"),
        }
    }
    let last_logits = {
        let all = preex.read_f32(&pre.g, pre.logits);
        all[((p - 1) * vocab) as usize..].to_vec()
    };

    let mut dec = parse(&model, 1, Some(CacheBind { cap: w.ctx, ctx: w.ctx }));
    let (dec_recipes, dec_dead) = if std::env::var("COGROUP").is_ok() {
        crate::cogroup::coarsen_cogroup(&mut dec.g)
    } else {
        (Vec::new(), Default::default())
    };
    let mut din = weights_as_inputs(&dec);
    add_concat(&mut din, &dec_recipes);
    din.insert("input_ids".into(), vec![w.gen_tokens[0] as f32]);
    din.insert("scatter_idx".into(), vec![p as f32]);
    din.insert("attn_mask".into(), mask_row(p, w.ctx));
    for id in &dec.cache_ins {
        let name = dec.input_nodes.iter().find(|(_, v)| **v == *id).map(|(n, _)| n.clone()).unwrap();
        din.insert(name, vec![0.0; (w.ctx * kvd_of(&dec, *id)) as usize]);
    }
    let mut decex = NativeExec::new();
    decex.inplace_cache = true;
    decex.dead = dec_dead;
    decex.build(&dec.g, &din);
    for (k, cin) in dec.cache_ins.iter().enumerate() {
        let vals = preex.read_f32(&pre.g, pre.cache_outs[k]);
        decex.poke_f32(*cin, &vals);
    }

    if std::env::var("DUMP_TAPE").is_ok() { decex.dump_tape(); }
    let ids_n = dec.input_nodes["input_ids"];
    let sc_n = dec.input_nodes["scatter_idx"];
    let mask_n = dec.input_nodes["attn_mask"];
    let mut first_logits = Vec::new();
    let mut final_logits = Vec::new();
    let t1 = Instant::now();
    for (t, tok) in w.gen_tokens.iter().enumerate() {
        let pos = p + t as i64;
        decex.poke_i32(ids_n, &[*tok]);
        decex.poke_i32(sc_n, &[pos as i32]);
        decex.poke_f32(mask_n, &mask_row(pos, w.ctx));
        decex.replay();
        if t == 0 {
            first_logits = decex.read_f32(&dec.g, dec.logits);
        }
        if t == g - 1 {
            final_logits = decex.read_f32(&dec.g, dec.logits);
        }
    }
    let gen_ms = t1.elapsed().as_secs_f64() * 1e3;

    SessionMetrics {
        ttft_ms: ttft,
        decode_tok_s: g as f64 / (gen_ms / 1e3),
        e2e_ms: ttft + gen_ms,
        prefill_dispatches: preex.dispatches,
        decode_dispatches: decex.dispatches,
        last_logits,
        first_logits,
        final_logits,
    }
}

fn run_session_mlx(w: &Workload, weights: &HashMap<String, Vec<f32>>, vocab: i64) -> SessionMetrics {
    use crate::mlx_ffi::MlxLlama;
    const D: i64 = 1024;
    const FF: i64 = 2752;
    const H: i64 = 16;
    const KV: i64 = 4;
    const LAYERS: i64 = 8;
    const EPS: f32 = 1e-5;
    let hd = D / H;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut m = MlxLlama::new(vocab as i32, D as i32, FF as i32, H as i32, KV as i32, LAYERS as i32, EPS, scale);
    m.set_weight("model.embed_tokens.weight", &weights["model.embed_tokens.weight"], &[vocab, D]);
    for l in 0..LAYERS {
        let mut set = |suf: &str, shape: [i64; 2]| {
            let name = format!("model.layers.{l}.{suf}.weight");
            m.set_weight(&name, &weights[&name], &shape);
        };
        set("self_attn.q_proj", [D, D]);
        set("self_attn.k_proj", [D, KV * hd]);
        set("self_attn.v_proj", [D, KV * hd]);
        set("self_attn.o_proj", [D, D]);
        set("mlp.gate_proj", [D, FF]);
        set("mlp.up_proj", [D, FF]);
        set("mlp.down_proj", [FF, D]);
    }
    let g = w.gen_tokens.len();
    let _ = m.prefill(&w.prompt);
    for &tok in &w.gen_tokens {
        let _ = m.decode(tok);
    }
    let t0 = Instant::now();
    let last = m.prefill(&w.prompt);
    let ttft = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let mut first = Vec::new();
    let mut final_ = Vec::new();
    for (t, &tok) in w.gen_tokens.iter().enumerate() {
        let lg = m.decode(tok);
        if t == 0 {
            first = lg.clone();
        }
        if t == g - 1 {
            final_ = lg;
        }
    }
    let gen_ms = t1.elapsed().as_secs_f64() * 1e3;
    SessionMetrics {
        ttft_ms: ttft,
        decode_tok_s: g as f64 / (gen_ms / 1e3),
        e2e_ms: ttft + gen_ms,
        prefill_dispatches: 0,
        decode_dispatches: 0,
        last_logits: last,
        first_logits: first,
        final_logits: final_,
    }
}

fn describe(g: &crate::ir::GraphIr, id: usize) -> String {
    use crate::ir::NodeKind;
    match &g.nodes[id].kind {
        NodeKind::Input { name } => format!("Input {name} dims={:?}", g.nodes[id].dims),
        NodeKind::LiteralF32(_) => "LitF32".into(),
        NodeKind::LiteralI32(_) => "LitI32".into(),
        NodeKind::Pointwise { nl, read } => format!("Pointwise {nl:?} <-{} dims={:?}", read.src, g.nodes[id].dims),
        NodeKind::Move { a, b, kext, pair, a_idx, write_idx, write_base, reduce_max } => format!(
            "Move k={kext} pair={pair} max={reduce_max} a<-{} b<-{:?} gather={} scatter={} dims={:?} a.nls={:?}",
            a.src, b.as_ref().map(|r| r.src), !a_idx.is_empty(), !write_idx.is_empty() || write_base.is_some(), g.nodes[id].dims, a.nls
        ),
    }
}

fn kvd_of(pm: &ParsedModel, cache_in: usize) -> i64 {
    *pm.g.nodes[cache_in].dims.last().unwrap()
}

fn resolve_vocab(vi: &crate::onnx_proto::ValueInfoProto) -> i64 {
    use crate::onnx_proto::tensor_shape_proto::dimension::Value as DV;
    use crate::onnx_proto::type_proto::Value;
    let Some(Value::TensorType(t)) = vi.r#type.as_ref().and_then(|t| t.value.as_ref()) else {
        panic!("logits type")
    };
    let dim = &t.shape.as_ref().unwrap().dim;
    match &dim[1].value {
        Some(DV::DimValue(v)) => *v,
        _ => panic!("vocab dim"),
    }
}
