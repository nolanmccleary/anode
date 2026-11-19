use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Device, Int, Tensor, TensorData};
use std::collections::HashMap;
use std::io::Write as _;
use std::time::Instant;

const VOCAB: usize = 32000;
const D: usize = 1024;
const H: usize = 16;
const KV: usize = 4;
const HD: usize = D / H;
const LAYERS: usize = 8;
const EPS: f32 = 1e-5;

fn load_weights(dir: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let mut w = HashMap::new();
    for line in std::fs::read_to_string(format!("{dir}/manifest.txt")).unwrap().lines() {
        let f: Vec<&str> = line.split('|').collect();
        if f[0] != "input" || f[1].starts_with("arg.") || f[2] != "F32" {
            continue;
        }
        let dims: Vec<usize> = f[3].split(',').map(|x| x.parse().unwrap()).collect();
        let raw = std::fs::read(format!("{dir}/{}", f[4])).unwrap();
        let vals: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        w.insert(f[1].to_string(), (dims, vals));
    }
    w
}

struct Layer {
    wq: Tensor<2>,
    wk: Tensor<2>,
    wv: Tensor<2>,
    wo: Tensor<2>,
    wg: Tensor<2>,
    wu: Tensor<2>,
    wd: Tensor<2>,
}

struct Model {
    embed: Tensor<2>,
    embed_t: Tensor<2>,
    layers: Vec<Layer>,
}

fn t2(w: &HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str, dev: &Device) -> Tensor<2> {
    let (dims, vals) = &w[name];
    assert_eq!(dims.len(), 2, "{name}");
    Tensor::from_data(TensorData::new(vals.clone(), [dims[0], dims[1]]), dev)
}

fn build(w: &HashMap<String, (Vec<usize>, Vec<f32>)>, dev: &Device) -> Model {
    let embed = t2(w, "model.embed_tokens.weight", dev);
    let embed_t = embed.clone().swap_dims(0, 1);
    let layers = (0..LAYERS)
        .map(|l| {
            let p = format!("model.layers.{l}");
            Layer {
                wq: t2(w, &format!("{p}.self_attn.q_proj.weight"), dev),
                wk: t2(w, &format!("{p}.self_attn.k_proj.weight"), dev),
                wv: t2(w, &format!("{p}.self_attn.v_proj.weight"), dev),
                wo: t2(w, &format!("{p}.self_attn.o_proj.weight"), dev),
                wg: t2(w, &format!("{p}.mlp.gate_proj.weight"), dev),
                wu: t2(w, &format!("{p}.mlp.up_proj.weight"), dev),
                wd: t2(w, &format!("{p}.mlp.down_proj.weight"), dev),
            }
        })
        .collect();
    Model { embed, embed_t, layers }
}

fn rms(x: &Tensor<2>) -> Tensor<2> {
    let denom = x.clone().mul(x.clone()).mean_dim(1).add_scalar(EPS).sqrt();
    x.clone().div(denom)
}

fn attn(q: Tensor<2>, kc: &Tensor<2>, vc: &Tensor<2>, mask: Option<&Tensor<2>>) -> Tensor<2> {
    let s = q.dims()[0];
    let t = kc.dims()[0];
    let g = H / KV;
    let qh = q.reshape([s, H, HD]).swap_dims(0, 1).reshape([KV, g * s, HD]);
    let kh = kc.clone().reshape([t, KV, HD]).swap_dims(0, 1);
    let vh = vc.clone().reshape([t, KV, HD]).swap_dims(0, 1);
    let mut scores = qh.matmul(kh.swap_dims(1, 2)).mul_scalar(1.0 / (HD as f32).sqrt());
    if let Some(m) = mask {
        scores = scores
            .reshape([KV, g, s, t])
            .add(m.clone().reshape([1, 1, s, t]))
            .reshape([KV, g * s, t]);
    }
    let o = softmax(scores, 2).matmul(vh);
    o.reshape([H, s, HD]).swap_dims(0, 1).reshape([s, D])
}

type Cache = Vec<(Tensor<2>, Tensor<2>)>;

fn forward(model: &Model, ids: &[i32], cache: Option<&Cache>, mask: Option<&Tensor<2>>, dev: &Device) -> (Tensor<2>, Cache) {
    let s = ids.len();
    let idx: Tensor<1, Int> = Tensor::from_data(TensorData::new(ids.to_vec(), [s]), dev);
    let mut x = model.embed.clone().select(0, idx);
    let mut new_cache = Vec::with_capacity(LAYERS);
    for (l, layer) in model.layers.iter().enumerate() {
        let nm = rms(&x);
        let q = nm.clone().matmul(layer.wq.clone());
        let k_new = nm.clone().matmul(layer.wk.clone());
        let v_new = nm.matmul(layer.wv.clone());
        let (kc, vc) = match cache {
            Some(c) => (
                Tensor::cat(vec![c[l].0.clone(), k_new], 0),
                Tensor::cat(vec![c[l].1.clone(), v_new], 0),
            ),
            None => (k_new, v_new),
        };
        x = x + attn(q, &kc, &vc, mask).matmul(layer.wo.clone());
        let nm2 = rms(&x);
        let gate = nm2.clone().matmul(layer.wg.clone());
        let up = nm2.matmul(layer.wu.clone());
        x = x + (gate.clone() * sigmoid(gate) * up).matmul(layer.wd.clone());
        new_cache.push((kc, vc));
    }
    (rms(&x).matmul(model.embed_t.clone()), new_cache)
}

fn causal_mask(s: usize, dev: &Device) -> Tensor<2> {
    let mut m = vec![0.0f32; s * s];
    for i in 0..s {
        for j in (i + 1)..s {
            m[i * s + j] = -1e10;
        }
    }
    Tensor::from_data(TensorData::new(m, [s, s]), dev)
}

fn last_row(logits: Tensor<2>) -> Vec<f32> {
    let s = logits.dims()[0];
    logits.slice([s - 1..s, 0..VOCAB]).into_data().try_to_vec().unwrap()
}

fn session(model: &Model, prompt: &[i32], gen_toks: &[i32], dev: &Device) -> (f64, f64, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mask = causal_mask(prompt.len(), dev);
    let t0 = Instant::now();
    let (logits_p, mut cache) = forward(model, prompt, None, Some(&mask), dev);
    let last = last_row(logits_p);
    let ttft = t0.elapsed().as_secs_f64() * 1e3;

    let mut first = Vec::new();
    let mut final_l = Vec::new();
    let t1 = Instant::now();
    for (t, tok) in gen_toks.iter().enumerate() {
        let (logits, nc) = forward(model, &[*tok], Some(&cache), None, dev);
        cache = nc;
        let row = last_row(logits);
        if t == 0 {
            first = row.clone();
        }
        if t == gen_toks.len() - 1 {
            final_l = row;
        }
    }
    let gen_ms = t1.elapsed().as_secs_f64() * 1e3;
    (ttft, gen_ms, last, first, final_l)
}

fn main() {
    let mut a = std::env::args().skip(1);
    let usage = "usage: run_burn <fatmodel dir> <p> <g> <c> <out_dir>";
    let dir = a.next().expect(usage);
    let p: usize = a.next().expect(usage).parse().unwrap();
    let g: usize = a.next().expect(usage).parse().unwrap();
    let _c: usize = a.next().expect(usage).parse().unwrap();
    let out_dir = a.next().expect(usage);

    let dev = Device::default();
    let w = load_weights(&dir);
    let model = build(&w, &dev);
    let prompt: Vec<i32> = (0..p as i32).map(|i| (i * 7 + 3) % VOCAB as i32).collect();
    let gen_toks: Vec<i32> = (0..g as i32).map(|t| (1000 + 13 * t) % VOCAB as i32).collect();

    session(&model, &prompt, &gen_toks, &dev);
    let (ttft, gen_ms, last, first, final_l) = session(&model, &prompt, &gen_toks, &dev);

    let bytes: Vec<u8> = last.iter().chain(&first).chain(&final_l).flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{out_dir}/burn_logits.bin"), bytes).unwrap();
    let mut o = std::io::stdout();
    writeln!(
        o,
        "{{\"name\":\"burn\",\"platform\":\"metal\",\"ttft_ms\":{:.3},\"decode_tok_s\":{:.3},\"e2e_ms\":{:.3},\"logits_file\":\"burn_logits.bin\"}}",
        ttft,
        g as f64 / (gen_ms / 1e3),
        ttft + gen_ms
    )
    .unwrap();
}
