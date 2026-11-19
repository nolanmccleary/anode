use crate::ir::{Dt, GraphIr, Nl, NodeKind, Read};
use cute_layout::Layout;
use crate::native_fuse;
use std::collections::HashMap;

fn leaves(t: &cute_layout::IntTuple) -> Vec<i64> {
    match t {
        cute_layout::IntTuple::Leaf(v) => vec![*v],
        cute_layout::IntTuple::Tuple(ts) => ts.iter().flat_map(leaves).collect(),
    }
}

fn axes_of(l: &Layout) -> (Vec<i64>, Vec<i64>) {
    (leaves(&l.shape), leaves(&l.stride))
}

fn eval_flat(sh: &[i64], st: &[i64], i: i64) -> i64 {
    let mut r = i;
    let mut off = 0i64;
    for k in (0..sh.len()).rev() {
        off += (r % sh[k]) * st[k];
        r /= sh[k];
    }
    off
}

fn numel(dims: &[i64]) -> i64 {
    dims.iter().product()
}

pub struct CpuRef {
    pub f: Vec<Option<Vec<f32>>>,
    pub i: Vec<Option<Vec<i32>>>,
}

impl CpuRef {
    fn fv(&self, id: usize) -> &[f32] {
        self.f[id].as_ref().expect("f32 value")
    }

    fn iv(&self, id: usize) -> &[i32] {
        self.i[id].as_ref().expect("i32 value")
    }

    fn read_val(&self, g: &GraphIr, r: &Read, idxs: &[(usize, Vec<i64>, Vec<i64>, i64)], i: i64) -> f32 {
        let (sh, st) = axes_of(&r.layout);
        let mut addr = eval_flat(&sh, &st, i);
        for (idx, ish, ist, stride) in idxs {
            addr += self.iv(*idx)[eval_flat(ish, ist, i) as usize] as i64 * stride;
        }
        match g.nodes[r.src].dtype {
            Dt::F32 => self.fv(r.src)[addr as usize],
            Dt::I32 => self.iv(r.src)[addr as usize] as f32,
        }
    }
}

pub fn eval(g: &GraphIr, inputs: &HashMap<String, Vec<f32>>) -> CpuRef {
    let mut m = CpuRef {
        f: vec![None; g.nodes.len()],
        i: vec![None; g.nodes.len()],
    };
    let fusions = if std::env::var("GENERAL_ONLY").is_ok() {
        native_fuse::Fusions {
            claims: Default::default(),
            rms: Default::default(),
            silu: Default::default(),
            attn: Default::default(),
        }
    } else {
        native_fuse::plan(g)
    };
    let litf = |g: &GraphIr, id: usize| -> f32 {
        match &g.nodes[id].kind {
            NodeKind::LiteralF32(v) => v[0],
            _ => panic!("expected scalar literal"),
        }
    };
    let mut node_sum = 0f64;
    for (id, node) in g.nodes.iter().enumerate() {
        let t_node = std::time::Instant::now();
        let n = numel(&node.dims);
        if fusions.claims.contains(&id) {
            node_sum += t_node.elapsed().as_secs_f64();
            continue;
        }
        if let Some((x, ce)) = fusions.rms.get(&id) {
            let xd = &g.nodes[*x].dims;
            let cols = *xd.last().unwrap() as usize;
            let rows = numel(xd) as usize / cols;
            let eps = litf(g, *ce);
            let xv = m.fv(*x).to_vec();
            let mut out = vec![0f32; rows * cols];
            for r in 0..rows {
                let mut ss = 0f32;
                for c in 0..cols {
                    ss += xv[r * cols + c] * xv[r * cols + c];
                }
                let rms = 1.0 / (ss / cols as f32 + eps).sqrt();
                for c in 0..cols {
                    out[r * cols + c] = xv[r * cols + c] * rms;
                }
            }
            m.f[id] = Some(out);
            node_sum += t_node.elapsed().as_secs_f64();
            continue;
        }
        if let Some((gg, up, cl, c1)) = fusions.silu.get(&id) {
            let clv = litf(g, *cl);
            let c1v = litf(g, *c1);
            let gv = m.fv(*gg).to_vec();
            let uv = m.fv(*up).to_vec();
            let out = gv
                .iter()
                .zip(&uv)
                .map(|(gx, ux)| gx / ((gx * clv).exp2() + c1v) * ux)
                .collect();
            m.f[id] = Some(out);
            node_sum += t_node.elapsed().as_secs_f64();
            continue;
        }
        if let Some((_q, _kc, _v, mask, cs, sc)) = fusions.attn.get(&id) {
            let NodeKind::Move { a: qa, b: Some(kb), kext: hd, .. } = &g.nodes[*sc].kind else {
                panic!("attn scores shape")
            };
            let NodeKind::Move { b: Some(vb), kext: ctxk, .. } = &g.nodes[id].kind else {
                panic!("attn pv shape")
            };
            let scd = g.nodes[*sc].dims.clone();
            let (kv, grp, sq, ctx) = (scd[0], scd[1], scd[2], scd[3]);
            assert_eq!(*ctxk, ctx, "attn ctx kext");
            let scale = litf(g, *cs);
            let maskv = m.fv(*mask).to_vec();
            let none: Vec<(usize, Vec<i64>, Vec<i64>, i64)> = Vec::new();
            let hdn = *hd;
            let mut out = vec![0f32; (kv * grp * sq * hdn) as usize];
            for b0 in 0..kv * grp * sq {
                let srow = b0 % sq;
                let mut xs = vec![0f32; ctx as usize];
                for c in 0..ctx {
                    let mut dot = 0f32;
                    for e in 0..hdn {
                        let i = (b0 * ctx + c) * hdn + e;
                        dot += m.read_val(g, qa, &none, i) * m.read_val(g, kb, &none, i);
                    }
                    xs[c as usize] = dot * scale + maskv[(srow * ctx + c) as usize];
                }
                let mx = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut l = 0f32;
                for x in xs.iter_mut() {
                    *x = (*x - mx).exp();
                    l += *x;
                }
                for e in 0..hdn {
                    let mut o = 0f32;
                    for c in 0..ctx {
                        let i = (b0 * hdn + e) * ctx + c;
                        o += xs[c as usize] * m.read_val(g, vb, &none, i);
                    }
                    out[(b0 * hdn + e) as usize] = o / l;
                }
            }
            m.f[id] = Some(out);
            node_sum += t_node.elapsed().as_secs_f64();
            continue;
        }
        match &node.kind {
            NodeKind::Input { name } => {
                let host = inputs.get(name).unwrap_or_else(|| panic!("missing input {name}"));
                assert_eq!(host.len() as i64, n, "input {name} size");
                match node.dtype {
                    Dt::F32 => m.f[id] = Some(host.clone()),
                    Dt::I32 => m.i[id] = Some(host.iter().map(|v| *v as i32).collect()),
                }
            }
            NodeKind::LiteralF32(v) => m.f[id] = Some(v.clone()),
            NodeKind::LiteralI32(v) => m.i[id] = Some(v.clone()),
            NodeKind::Pointwise { nl, read } => {
                let (sh, st) = axes_of(&read.layout);
                let mut out = vec![0f32; n as usize];
                for (c, o) in out.iter_mut().enumerate() {
                    let x = match g.nodes[read.src].dtype {
                        Dt::F32 => m.fv(read.src)[eval_flat(&sh, &st, c as i64) as usize],
                        Dt::I32 => m.iv(read.src)[eval_flat(&sh, &st, c as i64) as usize] as f32,
                    };
                    *o = match nl {
                        Nl::Exp2 => x.exp2(),
                        Nl::Recip => 1.0 / x,
                        Nl::Sqrt => x.sqrt(),
                        Nl::Rsqrt => 1.0 / x.sqrt(),
                        Nl::Relu => x.max(0.0),
                        Nl::Step => {
                            if x > 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        Nl::CastF32 => x,
                    };
                }
                m.f[id] = Some(out);
            }
            NodeKind::Move { a, b, kext, pair, a_idx, write_idx, write_base } => {
                assert_eq!(node.dtype, Dt::F32, "cpu ref: i32 move unsupported");
                let aix: Vec<(usize, Vec<i64>, Vec<i64>, i64)> = a_idx
                    .iter()
                    .map(|ir| {
                        let (ish, ist) = axes_of(&ir.idx_layout);
                        (ir.idx, ish, ist, ir.stride)
                    })
                    .collect();
                if !write_idx.is_empty() {
                    let base = write_base.expect("scatter base");
                    let mut out = m.fv(base).to_vec();
                    let src_n = numel(&g.nodes[a.src].dims);
                    let (wsh, _) = axes_of(&a.layout);
                    let mut wst = vec![0i64; wsh.len()];
                    if wsh.len() >= 2 {
                        wst[wsh.len() - 2] = 1;
                    }
                    let wix: Vec<(usize, Vec<i64>, Vec<i64>, i64)> = write_idx
                        .iter()
                        .map(|ir| {
                            let (ish, ist) = axes_of(&ir.idx_layout);
                            (ir.idx, ish, ist, ir.stride)
                        })
                        .collect();
                    for i in 0..src_n {
                        let val = m.read_val(g, a, &aix, i);
                        let mut waddr = eval_flat(&wsh, &wst, i);
                        for (idx, ish, ist, stride) in &wix {
                            waddr += m.iv(*idx)[eval_flat(ish, ist, i) as usize] as i64 * stride;
                        }
                        out[waddr as usize] = val;
                    }
                    m.f[id] = Some(out);
                } else {
                    let bix: Vec<(usize, Vec<i64>, Vec<i64>, i64)> = Vec::new();
                    let mut out = vec![0f32; n as usize];
                    for (c, o) in out.iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for k in 0..*kext {
                            let i = c as i64 * kext + k;
                            let av = m.read_val(g, a, &aix, i);
                            let term = match (b, pair) {
                                (Some(bb), true) => av * m.read_val(g, bb, &bix, i),
                                (Some(bb), false) => {
                                    if k == 0 {
                                        av
                                    } else {
                                        m.read_val(g, bb, &bix, i)
                                    }
                                }
                                (None, _) => av,
                            };
                            acc += term;
                        }
                        *o = acc;
                    }
                    m.f[id] = Some(out);
                }
            }
        }
        node_sum += t_node.elapsed().as_secs_f64();
    }
    eprintln!("cpu ref: per-node sum {:.1} ms", node_sum * 1e3);
    m
}
