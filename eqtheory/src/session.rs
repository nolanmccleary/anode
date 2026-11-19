use eqtheory::native_exec::NativeExec;
use eqtheory::native_model::{build, NativeCfg};
use std::collections::HashMap;
use std::time::Instant;

const VOCAB: i64 = 32_000;
const P: usize = 512;
const G: usize = 64;
const C: i64 = 640;

fn prompt_tokens() -> Vec<i32> {
    (0..P as i32).map(|i| (i * 7 + 3) % VOCAB as i32).collect()
}

fn gen_tokens() -> Vec<i32> {
    (0..G as i32).map(|t| (1000 + 13 * t) % VOCAB as i32).collect()
}

fn mask_row(pos: i64) -> Vec<f32> {
    (0..C).map(|c| if c > pos { -1e10 } else { 0.0 }).collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: session <fatmodel dir>");
    let mut inputs: HashMap<String, Vec<f32>> = HashMap::new();
    for line in std::fs::read_to_string(format!("{dir}/manifest.txt")).unwrap().lines() {
        let f: Vec<&str> = line.split('|').collect();
        if f[0] != "input" || f[1].starts_with("arg.") {
            continue;
        }
        let raw = std::fs::read(format!("{dir}/{}", f[4])).unwrap();
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        inputs.insert(f[1].to_string(), vals);
    }
    let layers = 8usize;
    let kvd = 256usize;
    for l in 0..layers {
        inputs.insert(format!("cache.{l}.k"), vec![0.0; C as usize * kvd]);
        inputs.insert(format!("cache.{l}.v"), vec![0.0; C as usize * kvd]);
    }

    let base = NativeCfg {
        vocab: VOCAB,
        d: 1024,
        ff: 2752,
        n_heads: 16,
        n_kv_heads: 4,
        layers: layers as i64,
        cap: C,
        ctx: C,
        seq: P as i64,
        prev: 0,
    };
    let prefill_model = build(&base, (0..C as i32).collect());
    let mut pre_inputs = inputs.clone();
    pre_inputs.insert("ids".into(), prompt_tokens().iter().map(|t| *t as f32).collect());
    pre_inputs.insert("scatter_idx".into(), (0..P as i32).map(|v| v as f32).collect());
    pre_inputs.insert("mask".into(), eqtheory::native_model::causal_mask(&base));
    let t0 = Instant::now();
    let mut pre = NativeExec::new();
    pre.build(&prefill_model.g, &pre_inputs);
    let pre_plan = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    pre.replay();
    let ttft = t1.elapsed().as_secs_f64() * 1e3;
    let last_logits: Vec<f32> = {
        let all = pre.read_f32(&prefill_model.g, prefill_model.logits.id);
        all[(P - 1) * VOCAB as usize..].to_vec()
    };
    println!(
        "prefill: {} dispatches ({} elided), plan {pre_plan:.1} ms, TTFT {ttft:.1} ms ({:.1} tok/s prompt)",
        pre.dispatches,
        pre.elided,
        P as f64 / (ttft / 1e3)
    );
    println!("prefill last-token logits[0..4]: {:?}", &last_logits[..4]);

    if std::env::var("SESSION_TIMED").is_ok() {
        pre.backend.begin_sampling(256);
        let (c1, g1) = pre.backend.sample_timestamps();
        pre.replay();
        let (c2, g2) = pre.backend.sample_timestamps();
        if let Some(ctx) = pre.backend.end_sampling() {
            let nsamp = ctx.labels.len();
            let ticks = ctx
                .buf
                .resolve_counter_range(metal::NSRange { location: 0, length: (nsamp * 2) as u64 });
            let scale = (c2 - c1) as f64 / (g2 - g1).max(1) as f64;
            let to_ms = |t: u64| t as f64 * scale / 1e6;
            let mut agg: HashMap<String, (usize, f64)> = HashMap::new();
            let mut gaps = 0f64;
            let mut overlap = 0f64;
            let mut ksum = 0f64;
            let mut first_start: Option<u64> = None;
            let mut prev_end: Option<u64> = None;
            for i in 0..nsamp {
                let s0 = ticks[2 * i];
                let e0 = ticks[2 * i + 1];
                if s0 == u64::MAX || e0 == u64::MAX || e0 <= s0 {
                    continue;
                }
                let d = to_ms(e0 - s0);
                ksum += d;
                let e = agg.entry(ctx.labels[i].clone()).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += d;
                if first_start.is_none() {
                    first_start = Some(s0);
                }
                if let Some(pe) = prev_end {
                    if s0 > pe {
                        gaps += to_ms(s0 - pe);
                    } else {
                        overlap += to_ms(pe - s0);
                    }
                }
                prev_end = Some(e0);
            }
            let mut rows: Vec<(String, (usize, f64))> = agg.into_iter().collect();
            rows.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap());
            for (name, (cnt, ms)) in &rows {
                println!("timed: {name:12} x{cnt:3} {ms:8.2} ms");
            }
            let span = to_ms(prev_end.unwrap_or(0) - first_start.unwrap_or(0));
            println!(
                "timed: kernels {ksum:.1} ms, gaps {gaps:.1} ms, overlap {overlap:.1} ms, span {span:.1} ms"
            );
        }
    }

    let dcfg = NativeCfg { seq: 1, prev: (P - 1) as i64, ..base };
    let dec_model = build(&dcfg, (0..C as i32).collect());
    let mut dec_inputs = inputs.clone();
    dec_inputs.insert("ids".into(), vec![gen_tokens()[0] as f32]);
    dec_inputs.insert("scatter_idx".into(), vec![P as f32]);
    dec_inputs.insert("mask".into(), mask_row((P - 1) as i64));
    let t2 = Instant::now();
    let mut dec = NativeExec::new();
    dec.inplace_cache = true;
    dec.build(&dec_model.g, &dec_inputs);
    let dec_plan = t2.elapsed().as_secs_f64() * 1e3;
    for l in 0..layers {
        let (kc, vc) = prefill_model.cache_outs[l];
        let kvals = pre.read_f32(&prefill_model.g, kc.id);
        let vvals = pre.read_f32(&prefill_model.g, vc.id);
        dec.poke_f32(dec_model.cache_ins[l].0.id, &kvals);
        dec.poke_f32(dec_model.cache_ins[l].1.id, &vvals);
    }

    let toks = gen_tokens();
    let mut first_logits: Vec<f32> = Vec::new();
    let mut final_logits: Vec<f32> = Vec::new();
    let mut best = f64::INFINITY;
    let t3 = Instant::now();
    if std::env::var("DUMP_TAPE").is_ok() { dec.dump_tape(); }
    for (t, tok) in toks.iter().enumerate() {
        let pos = (P + t) as i64;
        dec.poke_i32(dec_model.ids.id, &[*tok]);
        dec.poke_i32(dec_model.scatter_idx.id, &[pos as i32]);
        dec.poke_f32(dec_model.mask.id, &mask_row(pos));
        let ts = Instant::now();
        dec.replay();
        best = best.min(ts.elapsed().as_secs_f64() * 1e3);
        if t == 0 {
            first_logits = dec.read_f32(&dec_model.g, dec_model.logits.id);
        }
        if t == G - 1 {
            final_logits = dec.read_f32(&dec_model.g, dec_model.logits.id);
        }
    }
    let gen_ms = t3.elapsed().as_secs_f64() * 1e3;
    println!(
        "decode: plan {dec_plan:.1} ms, {} elided, {G} tokens in {gen_ms:.1} ms -> {:.0} tok/s (best step {best:.3} ms)",
        dec.elided,
        G as f64 / (gen_ms / 1e3)
    );
    println!(
        "end-to-end session (prompt {P} + gen {G}): {:.1} ms",
        ttft + gen_ms
    );
    println!("decode first logits[0..4]: {:?}", &first_logits[..4]);
    println!("decode final logits[0..4]: {:?}", &final_logits[..4]);
    std::fs::write(
        format!("{dir}/session_native.bin"),
        last_logits
            .iter()
            .chain(&first_logits)
            .chain(&final_logits)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
}
