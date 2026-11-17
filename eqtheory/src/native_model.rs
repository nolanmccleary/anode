use crate::ir::{identity, layout_of, rm_strides, Dt, GraphIr, Nl, Read, Tensor};

pub struct NativeCfg {
    pub vocab: i64,
    pub d: i64,
    pub ff: i64,
    pub n_heads: i64,
    pub n_kv_heads: i64,
    pub layers: i64,
    pub cap: i64,
    pub ctx: i64,
    pub seq: i64,
    pub prev: i64,
}

pub struct NativeModel {
    pub g: GraphIr,
    pub logits: Tensor,
    pub cache_outs: Vec<(Tensor, Tensor)>,
    pub ids: Tensor,
    pub gather_idx: Tensor,
    pub scatter_idx: Tensor,
    pub cache_ins: Vec<(Tensor, Tensor)>,
    pub mask: Tensor,
}

const LOG2E: f32 = std::f32::consts::LOG2_E;

fn rms_norm(g: &mut GraphIr, x: Tensor, d: i64, eps: f32) -> Tensor {
    let x2 = g.mul(x, x);
    let s = g.sum_last(x2);
    let ms = g.scale(s, 1.0 / d as f32);
    let e = g.lit_f32(vec![eps], vec![1]);
    let mse = g.add_bcast(ms, e);
    let r = g.pointwise(Nl::Rsqrt, mse);
    g.mul_bcast_row2(x, r)
}

fn silu(g: &mut GraphIr, x: Tensor) -> Tensor {
    let neg = g.scale(x, -LOG2E);
    let e = g.pointwise(Nl::Exp2, neg);
    let one = g.lit_f32(vec![1.0], vec![1]);
    let denom = g.add_bcast(e, one);
    let sig = g.pointwise(Nl::Recip, denom);
    g.mul(x, sig)
}

pub fn causal_mask(cfg: &NativeCfg) -> Vec<f32> {
    let (s, ctx) = (cfg.seq, cfg.ctx);
    let mut mask = vec![0f32; (s * ctx) as usize];
    for q in 0..s {
        for c in 0..ctx {
            if c > q + cfg.prev {
                mask[(q * ctx + c) as usize] = -1e10;
            }
        }
    }
    mask
}

pub fn build(cfg: &NativeCfg, gather_vals: Vec<i32>) -> NativeModel {
    let mut g = GraphIr::new();
    let (d, ff, s, ctx, cap) = (cfg.d, cfg.ff, cfg.seq, cfg.ctx, cfg.cap);
    let hd = d / cfg.n_heads;
    let kvd = cfg.n_kv_heads * hd;
    let (kv, grp) = (cfg.n_kv_heads, cfg.n_heads / cfg.n_kv_heads);
    let eps = 1e-5f32;
    let scale = 1.0 / (hd as f32).sqrt();

    let embed = g.input("model.embed_tokens.weight", vec![cfg.vocab, d], Dt::F32);
    let ids = g.input("ids", vec![s], Dt::I32);
    let gather_idx = g.lit_i32(gather_vals, vec![ctx]);
    let scatter_idx = g.input("scatter_idx", vec![s], Dt::I32);
    let mask_t = g.input("mask", vec![s, ctx], Dt::F32);

    let mut cache_ins = Vec::new();
    for l in 0..cfg.layers {
        let kc = g.input(&format!("cache.{l}.k"), vec![cap, kvd], Dt::F32);
        let vc = g.input(&format!("cache.{l}.v"), vec![cap, kvd], Dt::F32);
        cache_ins.push((kc, vc));
    }

    let mut x = g.gather_rows(embed, ids, s);
    let mut cache_outs = Vec::new();

    for l in 0..cfg.layers as usize {
        let p = format!("model.layers.{l}");
        let wq = g.input(&format!("{p}.self_attn.q_proj.weight"), vec![d, d], Dt::F32);
        let wk = g.input(&format!("{p}.self_attn.k_proj.weight"), vec![d, kvd], Dt::F32);
        let wv = g.input(&format!("{p}.self_attn.v_proj.weight"), vec![d, kvd], Dt::F32);
        let wo = g.input(&format!("{p}.self_attn.o_proj.weight"), vec![d, d], Dt::F32);
        let wgate = g.input(&format!("{p}.mlp.gate_proj.weight"), vec![d, ff], Dt::F32);
        let wup = g.input(&format!("{p}.mlp.up_proj.weight"), vec![d, ff], Dt::F32);
        let wdown = g.input(&format!("{p}.mlp.down_proj.weight"), vec![ff, d], Dt::F32);

        let normed = rms_norm(&mut g, x, d, eps);
        let q = g.matmul(normed, wq);
        let k = g.matmul(normed, wk);
        let v = g.matmul(normed, wv);

        let (kc_in, vc_in) = cache_ins[l];
        let kc = g.scatter_rows(kc_in, scatter_idx, k);
        let vc = g.scatter_rows(vc_in, scatter_idx, v);
        let keys = g.gather_rows(kc, gather_idx, ctx);
        let vals = g.gather_rows(vc, gather_idx, ctx);

        let sh = vec![kv, grp, s, ctx, hd];
        let qr = Read { src: q.id, layout: layout_of(&sh, &[grp * hd, hd, d, 0, 1]) };
        let kr = Read { src: keys.id, layout: layout_of(&sh, &[hd, 0, 0, kvd, 1]) };
        let scores_raw = g.movn(vec![kv, grp, s, ctx], qr, Some(kr), hd, true);
        let scores = g.scale(scores_raw, scale);

        let msh = vec![kv, grp, s, ctx, 2];
        let sa = {
            let mut st = rm_strides(&[kv, grp, s, ctx]);
            st.push(0);
            Read { src: scores.id, layout: layout_of(&msh, &st) }
        };
        let sb = Read { src: mask_t.id, layout: layout_of(&msh, &[0, 0, ctx, 1, 0]) };
        let biased = g.movn(vec![kv, grp, s, ctx], sa, Some(sb), 2, false);

        let scaled = g.scale(biased, LOG2E);
        let ex = g.pointwise(Nl::Exp2, scaled);
        let denom = g.sum_last(ex);
        let rden = g.pointwise(Nl::Recip, denom);
        let probs = g.mul_bcast_row2(ex, rden);

        let psh = vec![kv, grp, s, hd, ctx];
        let pr = Read {
            src: probs.id,
            layout: layout_of(&psh, &[grp * s * ctx, s * ctx, ctx, 0, 1]),
        };
        let vr = Read { src: vals.id, layout: layout_of(&psh, &[hd, 0, 0, 1, kvd]) };
        let attn_h = g.movn(vec![kv, grp, s, hd], pr, Some(vr), ctx, true);

        let flat = vec![s, d];
        let attn = if s == 1 {
            g.movn(flat.clone(), Read { src: attn_h.id, layout: layout_of(&[1, d, 1], &[0, 1, 0]) }, None, 1, true)
        } else {
            let perm = Read {
                src: attn_h.id,
                layout: layout_of(
                    &[s, kv, grp, hd, 1],
                    &[hd, grp * s * hd, s * hd, 1, 0],
                ),
            };
            g.movn(flat.clone(), perm, None, 1, true)
        };

        let o = g.matmul(attn, wo);
        x = g.add(x, o);

        let ffin = rms_norm(&mut g, x, d, eps);
        let gate = g.matmul(ffin, wgate);
        let up = g.matmul(ffin, wup);
        let act = silu(&mut g, gate);
        let h = g.mul(act, up);
        let down = g.matmul(h, wdown);
        x = g.add(x, down);

        cache_outs.push((kc, vc));
    }

    let fx = rms_norm(&mut g, x, d, eps);
    let lsh = vec![s, cfg.vocab, d];
    let fr = Read { src: fx.id, layout: layout_of(&lsh, &[d, 0, 1]) };
    let er = Read { src: embed.id, layout: layout_of(&lsh, &[0, d, 1]) };
    let logits = g.movn(vec![s, cfg.vocab], fr, Some(er), d, true);

    NativeModel {
        g,
        logits,
        cache_outs,
        ids,
        gather_idx,
        scatter_idx,
        cache_ins,
        mask: mask_t,
    }
}
