use crate::ir::{Dt, GraphIr, Nl, NodeKind, Read};
use cute_layout::Layout;
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
        let mut addr = eval_flat(&sh, &st, i) + r.base;
        for (idx, ish, ist, stride) in idxs {
            addr += self.iv(*idx)[eval_flat(ish, ist, i) as usize] as i64 * stride;
        }
        let val = match g.nodes[r.src].dtype {
            Dt::F32 => self.fv(r.src)[addr as usize],
            Dt::I32 => self.iv(r.src)[addr as usize] as f32,
        };
        apply_nls(val, &r.nls)
    }
}

fn apply_nls(mut x: f32, nls: &[Nl]) -> f32 {
    for nl in nls {
        x = match nl {
            Nl::Exp2 => x.exp2(),
            Nl::Recip => 1.0 / x,
            Nl::Sqrt => x.sqrt(),
            Nl::Rsqrt => 1.0 / x.sqrt(),
            Nl::Relu => x.max(0.0),
            Nl::Step => if x > 0.0 { 1.0 } else { 0.0 },
            Nl::CastF32 => x,
        };
    }
    x
}

pub fn eval(g: &GraphIr, inputs: &HashMap<String, Vec<f32>>) -> CpuRef {
    let mut m = CpuRef {
        f: vec![None; g.nodes.len()],
        i: vec![None; g.nodes.len()],
    };
    let mut node_sum = 0f64;
    for (id, node) in g.nodes.iter().enumerate() {
        let t_node = std::time::Instant::now();
        let n = numel(&node.dims);
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
                    let raw = match g.nodes[read.src].dtype {
                        Dt::F32 => m.fv(read.src)[eval_flat(&sh, &st, c as i64) as usize],
                        Dt::I32 => m.iv(read.src)[eval_flat(&sh, &st, c as i64) as usize] as f32,
                    };
                    let x = apply_nls(raw, &read.nls);
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
            NodeKind::Move { a, b, kext, pair, a_idx, write_idx, write_base, reduce_max } => {
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
                        let mut acc = if *reduce_max { f32::NEG_INFINITY } else { 0f32 };
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
                            if *reduce_max {
                                acc = acc.max(term);
                            } else {
                                acc += term;
                            }
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
