use eqtheory::affine_arm::AffineExec;
use eqtheory::arena::{allocate_arena, read_f32, upload};
use eqtheory::loaders;
use luminal_metal::bindings::MetalBindings;
use luminal_metal::MetalRuntime;
use std::time::Instant;

fn main() {
    luminal::logical_ir::set_cute_lowering(false);
    let loaded = loaders::load_default();
    let inputs = loaded.inputs();
    let read_ids = loaded.read_ids();
    let specs = loaded.graph.logical.input_specs();

    let mut rt = MetalRuntime::load_with(
        &loaded.graph,
        MetalBindings::dense(&loaded.graph.logical, &read_ids),
        eqtheory::kernels::shared_registry(),
    )
    .expect("load");
    let t0 = Instant::now();
    rt.compile(&Default::default(), &Default::default()).expect("compile");
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
    let plan = rt.plan().unwrap().clone();

    let t1 = Instant::now();
    let mut ax = AffineExec::new();
    ax.build(&plan, &specs, &inputs);
    let map_ms = t1.elapsed().as_secs_f64() * 1e3;
    if std::env::var("DUMP").is_ok() {
        ax.dump();
    }
    if std::env::var("HAZ").is_ok() {
        ax.hazards();
    }
    if std::env::var("VERIFY").is_ok() {
        ax.replay();
        ax.verify_tape();
    }
    ax.replay();
    let mut exec_ms = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        ax.replay();
        exec_ms = exec_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    if std::env::var("TRACE").is_ok() {
        ax.dump_outputs();
    }
    println!(
        "affine arm: {} dispatches ({} gemms, {} decomposed), peak planned {:.1} MB",
        ax.dispatches,
        ax.gemms,
        ax.decomposed_reduces,
        ax.peak_bytes as f64 / 1e6
    );
    println!(
        "luminal compile {compile_ms:.1} ms, map {map_ms:.1} ms, execute {exec_ms:.3} ms/token -> {:.0} tok/s",
        1000.0 / exec_ms
    );

    if std::env::var("DIAG").is_ok() {
        let find = |label: &str| -> Vec<f32> {
            let spec = specs.iter().find(|s| s.label.contains(label)).unwrap();
            inputs[&spec.id].as_f32().unwrap()
        };
        let embed = find("embed");
        let wk = find("layers.0.self_attn.k_proj");
        let wnorm = specs
            .iter()
            .find(|s| s.label.contains("layers.0") && s.label.contains("norm"))
            .map(|s| inputs[&s.id].as_f32().unwrap());
        let d = 512usize;
        let tok = 3usize;
        let x: Vec<f32> = embed[tok * d..(tok + 1) * d].to_vec();
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let r = 1.0 / (ms + 1e-5).sqrt();
        let normed: Vec<f32> = match &wnorm {
            Some(w) => x.iter().zip(w).map(|(v, g)| v * r * g).collect(),
            None => x.iter().map(|v| v * r).collect(),
        };
        let kdim = wk.len() / d;
        let mut krow = vec![0f32; kdim.min(4)];
        for o in 0..krow.len() {
            let mut acc = 0f32;
            for i in 0..d {
                acc += normed[i] * wk[o * d + i];
            }
            krow[o] = acc;
        }
        println!("host k[0..4] = {krow:?} (norm weight present: {})", wnorm.is_some());
        for cand in ["q_proj", "v_proj", "layers.1.self_attn.k_proj", "o_proj"] {
            if let Some(spec) = specs.iter().find(|s| s.label.contains(cand)) {
                let w = inputs[&spec.id].as_f32().unwrap();
                let outf = w.len() / d;
                let mut row = vec![0f32; 4.min(outf)];
                for o in 0..row.len() {
                    let mut acc = 0f32;
                    for i in 0..d {
                        acc += normed[i] * w[o * d + i];
                    }
                    row[o] = acc;
                }
                println!("host normed . {cand}^T [0..4] = {row:?}");
                let mut raw = vec![0f32; 4.min(outf)];
                for o in 0..raw.len() {
                    let mut acc = 0f32;
                    for i in 0..d {
                        acc += x[i] * w[o * d + i];
                    }
                    raw[o] = acc;
                }
                println!("host raw-x . {cand}^T [0..4]  = {raw:?}");
            }
        }
        let mut rawk = vec![0f32; 4];
        for o in 0..4 {
            let mut acc = 0f32;
            for i in 0..d {
                acc += x[i] * wk[o * d + i];
            }
            rawk[o] = acc;
        }
        println!("host raw-x . k_proj^T [0..4]  = {rawk:?}");
    }
    if std::env::var("DIAG2").is_ok() {
        let spec = specs.iter().find(|s| s.label.contains("layers.0.self_attn.k_proj")).unwrap();
        let w = inputs[&spec.id].as_f32().unwrap();
        println!("Wk flat[0..6]          = {:?}", &w[0..6]);
        println!("Wk col0 (W[i*128])     = {:?}", (0..6).map(|i| w[i * 128]).collect::<Vec<_>>());
        println!("buf101[0..6]           = {:?}", ax.read_alloc(101, 6));
        println!("buf102[0..4] (k out)   = {:?}", ax.read_alloc(102, 4));
        let d = 512usize;
        let embed = {
            let es = specs.iter().find(|s| s.label.contains("embed")).unwrap();
            inputs[&es.id].as_f32().unwrap()
        };
        let x: Vec<f32> = embed[3 * d..4 * d].to_vec();
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let r = 1.0 / (ms + 1e-5).sqrt();
        let normed: Vec<f32> = x.iter().map(|v| v * r).collect();
        let kin: Vec<f32> = (0..4)
            .map(|o| (0..d).map(|i| normed[i] * w[i * 128 + o]).sum())
            .collect();
        println!("host (in,out) k[0..4]  = {kin:?}");
    }
    let mut arena = allocate_arena(&rt).unwrap();
    for (id, hb) in inputs.iter() {
        upload(&rt, &mut arena, *id, hb.clone()).unwrap();
    }
    rt.execute(arena.buffer()).unwrap();

    for (k, out) in loaded.outputs.iter().enumerate() {
        let ours = ax.read_output(k);
        let oracle = read_f32(&rt, &arena, out.id).unwrap();
        assert_eq!(ours.len(), oracle.len(), "output {k} length");
        let peak = oracle.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));
        let d = ours.iter().zip(&oracle).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let ok = if peak > 0.0 { d / peak < 1e-4 } else { d == 0.0 };
        if !ok && std::env::var("VALS").is_ok() {
            let at = ours.iter().zip(&oracle).position(|(a, b)| (a - b).abs() > 1e-3).unwrap_or(0);
            println!(
                "  first diff at {at}: ours {:?} oracle {:?}",
                &ours[at..(at + 4).min(ours.len())],
                &oracle[at..(at + 4).min(oracle.len())]
            );
        }
        println!(
            "output {k}: max abs diff {d:.3e} (peak {peak:.2e}) -> {}",
            if ok { "MATCH" } else { "MISMATCH" }
        );
    }
}
