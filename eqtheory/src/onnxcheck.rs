use eqtheory::native_exec::NativeExec;
use eqtheory::native_model::{build, NativeCfg};
use eqtheory::onnx_import;
use std::collections::HashMap;

const VOCAB: i64 = 32000;
const S: i64 = 8;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: onnxcheck <fatmodel dir> <model.onnx>");
    let onnx_path = std::env::args().nth(2).expect("onnx path");

    let mut fat: HashMap<String, Vec<f32>> = HashMap::new();
    for line in std::fs::read_to_string(format!("{dir}/manifest.txt")).unwrap().lines() {
        let f: Vec<&str> = line.split('|').collect();
        if f[0] != "input" || f[1].starts_with("arg.") {
            continue;
        }
        let raw = std::fs::read(format!("{dir}/{}", f[4])).unwrap();
        fat.insert(f[1].to_string(), raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }

    let ids: Vec<f32> = (0..S).map(|i| ((i * 7 + 3) % VOCAB) as f32).collect();
    let mut mask = vec![0f32; (S * S) as usize];
    for q in 0..S {
        for c in 0..S {
            if c > q {
                mask[(q * S + c) as usize] = -1e10;
            }
        }
    }

    let cfg = NativeCfg {
        vocab: VOCAB, d: 1024, ff: 2752, n_heads: 16, n_kv_heads: 4, layers: 8,
        cap: S, ctx: S, seq: S, prev: 0,
    };
    let nm = build(&cfg, (0..S as i32).collect());
    let mut ninputs = fat.clone();
    ninputs.insert("ids".into(), ids.clone());
    ninputs.insert("scatter_idx".into(), (0..S).map(|i| i as f32).collect());
    ninputs.insert("mask".into(), mask.clone());
    for l in 0..8 {
        ninputs.insert(format!("cache.{l}.k"), vec![0.0; (S * 256) as usize]);
        ninputs.insert(format!("cache.{l}.v"), vec![0.0; (S * 256) as usize]);
    }
    let mut nex = NativeExec::new();
    nex.build(&nm.g, &ninputs);
    nex.replay();
    let nlog = nex.read_f32(&nm.g, nm.logits.id);

    let bytes = std::fs::read(&onnx_path).unwrap();
    let parsed = onnx_import::parse_bytes(&bytes, S);
    let logits_id = parsed.outputs.iter().find(|(n, _)| n == "logits").unwrap().1;
    let mut oinputs = parsed.weights.clone();
    oinputs.insert("input_ids".into(), ids.clone());
    oinputs.insert("attn_mask".into(), mask.clone());
    let mut oex = NativeExec::new();
    oex.build(&parsed.g, &oinputs);
    oex.replay();
    let olog = oex.read_f32(&parsed.g, logits_id);

    println!("native graph {} nodes, onnx graph {} nodes", nm.g.nodes.len(), parsed.g.nodes.len());
    println!("runtime inputs: {:?}", parsed.runtime_inputs.iter().map(|(n, s, _)| (n.clone(), s.clone())).collect::<Vec<_>>());
    let last = ((S - 1) * VOCAB) as usize;
    let (nrow, orow) = (&nlog[last..], &olog[last..]);
    let mut mx = 0f32;
    let mut peak = 0f32;
    for (a, b) in nrow.iter().zip(orow.iter()) {
        mx = mx.max((a - b).abs());
        peak = peak.max(a.abs());
    }
    println!("last-token logits: max abs diff {mx:.3e} (peak {peak:.2e}, rel {:.2e}) -> {}", mx / peak, if mx <= 1e-3 * peak { "MATCH" } else { "MISMATCH" });
    println!("onnx logits[0..4]: {:?}", &orow[..4]);
    println!("native logits[0..4]: {:?}", &nrow[..4]);

    let c = 8i64;
    let dcfg = NativeCfg { cap: c, ctx: c, seq: 1, prev: 0, ..cfg };
    let dnm = build(&dcfg, (0..c as i32).collect());
    let mut dmask = vec![0f32; c as usize];
    for j in 1..c {
        dmask[j as usize] = -1e10;
    }
    let mut dnin = fat.clone();
    dnin.insert("ids".into(), vec![3.0]);
    dnin.insert("scatter_idx".into(), vec![0.0]);
    dnin.insert("mask".into(), dmask.clone());
    for l in 0..8 {
        dnin.insert(format!("cache.{l}.k"), vec![0.0; (c * 256) as usize]);
        dnin.insert(format!("cache.{l}.v"), vec![0.0; (c * 256) as usize]);
    }
    let mut dnex = NativeExec::new();
    dnex.build(&dnm.g, &dnin);
    dnex.replay();
    let dnlog = dnex.read_f32(&dnm.g, dnm.logits.id);

    let dparsed = onnx_import::parse(
        &<eqtheory::onnx_proto::ModelProto as prost::Message>::decode(&bytes[..]).unwrap(),
        1,
        Some(onnx_import::CacheBind { cap: c, ctx: c }),
    );
    let mut doin = dparsed.weights.clone();
    doin.insert("input_ids".into(), vec![3.0]);
    doin.insert("attn_mask".into(), dmask.clone());
    doin.insert("scatter_idx".into(), vec![0.0]);
    for l in 0..8 {
        doin.insert(format!("past_k_{l}"), vec![0.0; (c * 256) as usize]);
        doin.insert(format!("past_v_{l}"), vec![0.0; (c * 256) as usize]);
    }
    let mut doex = NativeExec::new();
    doex.build(&dparsed.g, &doin);
    doex.replay();
    let dolog = doex.read_f32(&dparsed.g, dparsed.logits);
    let mut dmx = 0f32;
    let mut dpeak = 0f32;
    for (a, b) in dnlog.iter().zip(dolog.iter()) {
        dmx = dmx.max((a - b).abs());
        dpeak = dpeak.max(a.abs());
    }
    println!(
        "DECODE last-token logits: max abs diff {dmx:.3e} (peak {dpeak:.2e}) -> {}",
        if dmx <= 1e-3 * dpeak { "MATCH" } else { "MISMATCH" }
    );
    println!("decode onnx[0..4]: {:?}", &dolog[..4]);
    println!("decode native[0..4]: {:?}", &dnlog[..4]);
}
