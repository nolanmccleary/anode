use eqtheory_baseline::executor::Exec;
use eqtheory_baseline::loaders::{load_session, ModelConfig};
use luminal_metal::HostBuffer;
use std::collections::HashMap;
use std::io::Write as _;
use std::time::Instant;

const VOCAB: i64 = 32000;

fn fatmodel_weights(dir: &str) -> HashMap<String, Vec<f32>> {
    let mut w = HashMap::new();
    for line in std::fs::read_to_string(format!("{dir}/manifest.txt")).unwrap().lines() {
        let f: Vec<&str> = line.split('|').collect();
        if f[0] != "input" || f[1].starts_with("arg.") {
            continue;
        }
        let raw = std::fs::read(format!("{dir}/{}", f[4])).unwrap();
        w.insert(f[1].to_string(), raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }
    w
}

fn cfg(c: i64) -> ModelConfig {
    ModelConfig { vocab: VOCAB as usize, d: 1024, ff: 2752, n_heads: 16, n_kv_heads: 4, layers: 8, cache_capacity: c as usize }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let usage = "usage: run_luminal <fatmodel dir> <p> <g> <c> <out_dir>";
    let dir = a.next().expect(usage);
    let p: i64 = a.next().expect(usage).parse().unwrap();
    let g: i64 = a.next().expect(usage).parse().unwrap();
    let c: i64 = a.next().expect(usage).parse().unwrap();
    let out_dir = a.next().expect(usage);
    luminal::logical_ir::set_cute_lowering(false);
    let w = fatmodel_weights(&dir);
    let prompt: Vec<i32> = (0..p as i32).map(|i| (i * 7 + 3) % VOCAB as i32).collect();
    let gen_toks: Vec<i32> = (0..g as i32).map(|t| (1000 + 13 * t) % VOCAB as i32).collect();

    let pre = load_session(&cfg(c), p as usize, c as usize, prompt.clone());
    let mut pin = pre.loaded.inputs();
    for spec in pre.loaded.graph.logical.input_specs() {
        if let Some(vals) = w.get(&spec.label) {
            pin.insert(spec.id, HostBuffer::from(vals.clone()));
        }
    }
    let mut pex = Exec::new(pre.loaded.graph.logical.petgraph());
    let tr = Instant::now();
    pex.run(&pin);
    eprintln!("prompt run: {:.1}ms", tr.elapsed().as_secs_f64() * 1e3);
    let t0 = Instant::now();
    pex.replay();
    let ttft = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!("prompt replay: {ttft:.1}ms");
    let logits_p = pex.read_f32(pre.loaded.outputs[0].id);
    let last: Vec<f32> = logits_p[((p - 1) * VOCAB) as usize..].to_vec();
    let caches: Vec<Vec<f32>> = pre.loaded.outputs[1..].iter().map(|o| pex.read_f32(o.id)).collect();

    let dec = load_session(&cfg(c), 1, c as usize, vec![gen_toks[0]]);
    let mut din = dec.loaded.inputs();
    for spec in dec.loaded.graph.logical.input_specs() {
        if let Some(vals) = w.get(&spec.label) {
            din.insert(spec.id, HostBuffer::from(vals.clone()));
        }
    }
    let cache_ids: Vec<_> = dec
        .loaded
        .graph
        .logical
        .input_specs()
        .into_iter()
        .filter(|s| {
            let d: Vec<usize> = s.dims.iter().map(|x| x.to_usize().unwrap()).collect();
            d == vec![c as usize, 256]
        })
        .map(|s| s.id)
        .collect();
    let mut dex = Exec::new(dec.loaded.graph.logical.petgraph());
    dex.inplace_cache = true;
    dex.run(&din);
    for (k, id) in cache_ids.iter().enumerate() {
        dex.poke_f32(*id, &caches[k]);
    }
    let pos_iotas = dex.find_zero_scalar_iotas();

    let mut first = Vec::new();
    let mut final_l = Vec::new();
    let t1 = Instant::now();
    for (t, tok) in gen_toks.iter().enumerate() {
        let pos = p + t as i64;
        dex.poke_i32(dec.ids_id, &[*tok]);
        dex.poke_i32(dec.scatter_id, &[pos as i32]);
        for it in &pos_iotas {
            dex.poke_i32(*it, &[pos as i32]);
        }
        dex.replay();
        if t == 0 {
            first = dex.read_f32(dec.loaded.outputs[0].id);
        }
        if t == g as usize - 1 {
            final_l = dex.read_f32(dec.loaded.outputs[0].id);
        }
    }
    let gen_ms = t1.elapsed().as_secs_f64() * 1e3;

    let bytes: Vec<u8> = last.iter().chain(&first).chain(&final_l).flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{out_dir}/luminal_logits.bin"), bytes).unwrap();
    let mut o = std::io::stdout();
    writeln!(
        o,
        "{{\"name\":\"luminal\",\"platform\":\"metal\",\"ttft_ms\":{:.3},\"decode_tok_s\":{:.3},\"e2e_ms\":{:.3},\"logits_file\":\"luminal_logits.bin\"}}",
        ttft, g as f64 / (gen_ms / 1e3), ttft + gen_ms
    ).unwrap();
}
