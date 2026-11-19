use eqtheory::native_exec::NativeExec;
use eqtheory::native_model::{build, NativeCfg};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: nativerun <fatmodel dir>");
    let mut inputs: HashMap<String, Vec<f32>> = HashMap::new();
    let mut args_order: Vec<Vec<f32>> = Vec::new();
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for line in std::fs::read_to_string(format!("{dir}/manifest.txt")).unwrap().lines() {
        let f: Vec<&str> = line.split('|').collect();
        let raw = std::fs::read(format!("{dir}/{}", f[4])).unwrap();
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        match f[0] {
            "input" => {
                if f[1].starts_with("arg.") {
                    args_order.push(vals);
                } else {
                    inputs.insert(f[1].to_string(), vals);
                }
            }
            _ => outputs.push(vals),
        }
    }
    let layers = 8usize;
    assert_eq!(args_order.len(), 1 + 2 * layers + 2, "anon arg count");
    for l in 0..layers {
        inputs.insert(format!("cache.{l}.k"), args_order[1 + 2 * l].clone());
        inputs.insert(format!("cache.{l}.v"), args_order[2 + 2 * l].clone());
    }

    let cfg = NativeCfg {
        vocab: 32_000,
        d: 1024,
        ff: 2752,
        n_heads: 16,
        n_kv_heads: 4,
        layers: layers as i64,
        cap: 4,
        ctx: 2,
        seq: 1,
        prev: 1,
    };
    inputs.insert("ids".into(), vec![3.0]);
    inputs.insert("scatter_idx".into(), vec![1.0]);
    inputs.insert("mask".into(), eqtheory::native_model::causal_mask(&cfg));
    let model = build(&cfg, vec![0, 1]);
    println!("native graph: {} nodes", model.g.nodes.len());

    if std::env::var("CPU_REF").is_ok() {
        let t0 = Instant::now();
        let vals = eqtheory::cpu_ref::eval(&model.g, &inputs);
        println!("cpu ref: evaluated in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        let mut order: Vec<usize> = vec![model.logits.id];
        for (kc, vc) in &model.cache_outs {
            order.push(kc.id);
            order.push(vc.id);
        }
        let mut all_ok = true;
        for (k, (id, want)) in order.iter().zip(outputs.iter()).enumerate() {
            let got = vals.f[*id].as_ref().expect("cpu value");
            let mut mx = 0f32;
            let mut peak = 0f32;
            for (a, b) in got.iter().zip(want.iter()) {
                mx = mx.max((a - b).abs());
                peak = peak.max(b.abs());
            }
            let ok = mx <= 1e-3 * peak.max(1.0);
            all_ok &= ok;
            if k == 0 || !ok {
                println!(
                    "cpu output {k}: max abs diff {mx:.3e} (peak {peak:.2e}) -> {}",
                    if ok { "MATCH" } else { "MISMATCH" }
                );
            }
        }
        println!(
            "cpu ref vs exported oracle: {}",
            if all_ok { "ALL MATCH" } else { "MISMATCH" }
        );
        return;
    }

    let t0 = Instant::now();
    let mut ex = NativeExec::new();
    ex.build(&model.g, &inputs);
    let plan_ms = t0.elapsed().as_secs_f64() * 1e3;
    ex.replay();
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        ex.replay();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "native exec: {} dispatches, plan {plan_ms:.1} ms, execute {best:.3} ms/token -> {:.0} tok/s",
        ex.dispatches,
        1000.0 / best
    );

    let mut outs = vec![model.logits.id];
    for (k, v) in &model.cache_outs {
        outs.push(k.id);
        outs.push(v.id);
    }
    let mut all_ok = true;
    for (i, id) in outs.iter().enumerate() {
        let got = ex.read_f32(&model.g, *id);
        let refv = &outputs[i];
        assert_eq!(got.len(), refv.len(), "output {i} len");
        let peak = refv.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));
        let d = got.iter().zip(refv).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let ok = d / peak < 1e-4;
        all_ok &= ok;
        if i == 0 || !ok {
            println!("output {i}: max abs diff {d:.3e} (peak {peak:.2e}) -> {}", if ok { "MATCH" } else { "MISMATCH" });
        }
    }
    println!("all 17 outputs vs exported oracle: {}", if all_ok { "MATCH" } else { "MISMATCH" });
}
