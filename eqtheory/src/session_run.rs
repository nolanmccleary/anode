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

pub fn run_session(onnx_bytes: &[u8], w: &Workload) -> SessionMetrics {
    let model = ModelProto::decode(onnx_bytes).expect("decode onnx");
    let p = w.prompt.len() as i64;
    let g = w.gen_tokens.len();
    let vocab = {
        let out = model.graph.as_ref().unwrap().output.iter().find(|o| o.name() == "logits").unwrap();
        resolve_vocab(out)
    };

    let pre = parse(&model, p, Some(CacheBind { cap: w.ctx, ctx: w.ctx }));
    let mut pin = weights_as_inputs(&pre);
    pin.insert("input_ids".into(), w.prompt.iter().map(|t| *t as f32).collect());
    pin.insert("scatter_idx".into(), (0..p).map(|i| i as f32).collect());
    pin.insert("attn_mask".into(), causal(p, w.ctx));
    for id in &pre.cache_ins {
        let name = pre.input_nodes.iter().find(|(_, v)| **v == *id).map(|(n, _)| n.clone()).unwrap();
        pin.insert(name, vec![0.0; (w.ctx * kvd_of(&pre, *id)) as usize]);
    }
    let mut preex = NativeExec::new();
    preex.build(&pre.g, &pin);
    let t0 = Instant::now();
    preex.replay();
    let ttft = t0.elapsed().as_secs_f64() * 1e3;
    let last_logits = {
        let all = preex.read_f32(&pre.g, pre.logits);
        all[((p - 1) * vocab) as usize..].to_vec()
    };

    let dec = parse(&model, 1, Some(CacheBind { cap: w.ctx, ctx: w.ctx }));
    let mut din = weights_as_inputs(&dec);
    din.insert("input_ids".into(), vec![w.gen_tokens[0] as f32]);
    din.insert("scatter_idx".into(), vec![p as f32]);
    din.insert("attn_mask".into(), mask_row(p, w.ctx));
    for id in &dec.cache_ins {
        let name = dec.input_nodes.iter().find(|(_, v)| **v == *id).map(|(n, _)| n.clone()).unwrap();
        din.insert(name, vec![0.0; (w.ctx * kvd_of(&dec, *id)) as usize]);
    }
    let mut decex = NativeExec::new();
    decex.inplace_cache = true;
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
