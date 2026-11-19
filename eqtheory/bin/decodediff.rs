use eqtheory::native_exec::NativeExec;
use eqtheory::native_model::{build, NativeCfg};
use eqtheory::onnx_import::{self, CacheBind};
use std::collections::HashMap;

const C: i64 = 8;

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let onnx_path = std::env::args().nth(2).unwrap();
    let mut fat: HashMap<String, Vec<f32>> = HashMap::new();
    for line in std::fs::read_to_string(format!("{dir}/manifest.txt")).unwrap().lines() {
        let f: Vec<&str> = line.split('|').collect();
        if f[0] != "input" || f[1].starts_with("arg.") {
            continue;
        }
        let raw = std::fs::read(format!("{dir}/{}", f[4])).unwrap();
        fat.insert(f[1].to_string(), raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }
    let mut mask = vec![0f32; C as usize];
    for j in 1..C {
        mask[j as usize] = -1e10;
    }

    let cfg = NativeCfg { vocab: 32000, d: 1024, ff: 2752, n_heads: 16, n_kv_heads: 4, layers: 8, cap: C, ctx: C, seq: 1, prev: 0 };
    let nm = build(&cfg, (0..C as i32).collect());
    let mut nin = fat.clone();
    nin.insert("ids".into(), vec![3.0]);
    nin.insert("scatter_idx".into(), vec![0.0]);
    nin.insert("mask".into(), mask.clone());
    for l in 0..8 {
        nin.insert(format!("cache.{l}.k"), vec![0.0; (C * 256) as usize]);
        nin.insert(format!("cache.{l}.v"), vec![0.0; (C * 256) as usize]);
    }
    let mut nex = NativeExec::new();
    nex.build(&nm.g, &nin);
    nex.replay();

    let bytes = std::fs::read(&onnx_path).unwrap();
    let pm = onnx_import::parse(
        &<eqtheory::onnx_proto::ModelProto as prost::Message>::decode(&bytes[..]).unwrap(),
        1,
        Some(CacheBind { cap: C, ctx: C }),
    );
    let mut oin = pm.weights.clone();
    oin.insert("input_ids".into(), vec![3.0]);
    oin.insert("attn_mask".into(), mask.clone());
    oin.insert("scatter_idx".into(), vec![0.0]);
    for l in 0..8 {
        oin.insert(format!("past_k_{l}"), vec![0.0; (C * 256) as usize]);
        oin.insert(format!("past_v_{l}"), vec![0.0; (C * 256) as usize]);
    }
    let mut oex = NativeExec::new();
    oex.build(&pm.g, &oin);
    oex.replay();

    let omap: HashMap<String, usize> = pm.probes.iter().cloned().collect();
    println!("{:<16}{:>14}{:>14}  {}", "probe", "max_abs_diff", "peak", "verdict");
    for (label, nid) in &nm.probes {
        let Some(oid) = omap.get(label) else { continue };
        let nv = nex.read_f32(&nm.g, *nid);
        let ov = oex.read_f32(&pm.g, *oid);
        let mut mx = 0f32;
        let mut peak = 0f32;
        for (a, b) in nv.iter().zip(ov.iter()) {
            mx = mx.max((a - b).abs());
            peak = peak.max(a.abs());
        }
        let verdict = if mx == 0.0 { "IDENTICAL" } else { "DIFFERS" };
        println!("{label:<16}{mx:>14.3e}{peak:>14.3e}  {verdict}");
    }
}
