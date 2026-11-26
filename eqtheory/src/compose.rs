use crate::ir::{GraphIr, IdxRead, Nl, Node, NodeKind, Read};
use cute_layout::algebra::composition;
use cute_layout::{IntTuple, Layout};
use std::collections::HashMap;

fn leaves(t: &IntTuple) -> Vec<i64> {
    match t {
        IntTuple::Leaf(v) => vec![*v],
        IntTuple::Tuple(ts) => ts.iter().flat_map(leaves).collect(),
    }
}

fn flat_rev(l: &Layout) -> Layout {
    let sh: Vec<IntTuple> = leaves(&l.shape).into_iter().rev().map(IntTuple::Leaf).collect();
    let st: Vec<IntTuple> = leaves(&l.stride).into_iter().rev().map(IntTuple::Leaf).collect();
    Layout::new(IntTuple::Tuple(sh), IntTuple::Tuple(st))
}

pub fn rowmajor_compose(outer: &Layout, inner: &Layout) -> Result<Layout, String> {
    let rc = match composition(&flat_rev(outer), &flat_rev(inner)) {
        Ok(c) => flat_rev(&c),
        Err(e) => return Err(e.to_string()),
    };
    let n = inner.size();
    if rc.size() != n {
        return Err("composed domain size differs from consumer".into());
    }
    let osize = outer.size();
    for i in 0..n {
        let bi = eval_at(inner, i);
        if bi < 0 || bi >= osize {
            return Err("consumer read escapes producer domain".into());
        }
        if eval_at(&rc, i) != eval_at(outer, bi) {
            return Err("composition not access-preserving (non-nested outer): cute composition returned Ok but disagrees with the chained access".into());
        }
    }
    Ok(rc)
}

pub fn eval_at(l: &Layout, i: i64) -> i64 {
    let sh = leaves(&l.shape);
    let st = leaves(&l.stride);
    let mut r = i;
    let mut off = 0i64;
    for k in (0..sh.len()).rev() {
        off += (r % sh[k]) * st[k];
        r /= sh[k];
    }
    off
}

#[derive(Clone, Debug)]
pub struct Fused {
    pub layout: Layout,
    pub base: i64,
}

pub fn edge_compose(producer: &Read, consumer: &Read) -> Result<Fused, String> {
    if producer.base != 0 || consumer.base != 0 {
        return Err("nonzero Read.base not supported: composition is linear and the executor (cpu_ref) does not honor base".into());
    }
    let layout = rowmajor_compose(&producer.layout, &consumer.layout)?;
    Ok(Fused { layout, base: 0 })
}

pub fn is_view_or_pointwise(k: &NodeKind) -> Option<&Read> {
    match k {
        NodeKind::Pointwise { read, .. } => Some(read),
        NodeKind::Move { a, b: None, kext, a_idx, write_idx, write_base, .. }
            if *kext == 1 && a_idx.is_empty() && write_idx.is_empty() && write_base.is_none() =>
        {
            Some(a)
        }
        _ => None,
    }
}

fn kind_tag(k: &NodeKind) -> String {
    match k {
        NodeKind::Input { name } => format!("Input {name}"),
        NodeKind::LiteralF32(_) => "LitF32".into(),
        NodeKind::LiteralI32(_) => "LitI32".into(),
        NodeKind::Pointwise { nl, .. } => format!("Pointwise {nl:?}"),
        NodeKind::Move { a: _, b, kext, pair, a_idx, write_idx, write_base, reduce_max } => {
            let kind = if b.is_some() && *pair && *kext > 1 {
                "contract"
            } else if b.is_some() && *pair {
                "mul"
            } else if b.is_some() {
                "add"
            } else if *kext > 1 {
                if *reduce_max { "reduce-max" } else { "reduce" }
            } else {
                "view"
            };
            let g = if !a_idx.is_empty() { "+gather" } else { "" };
            let s = if !write_idx.is_empty() || write_base.is_some() { "+scatter" } else { "" };
            format!("Move/{kind}{g}{s} k={kext}")
        }
    }
}

pub fn reads_of(k: &NodeKind) -> Vec<&Read> {
    match k {
        NodeKind::Pointwise { read, .. } => vec![read],
        NodeKind::Move { a, b, .. } => match b {
            Some(b) => vec![a, b],
            None => vec![a],
        },
        _ => vec![],
    }
}

#[derive(Clone, Debug)]
pub struct Collapsed {
    pub src: usize,
    pub layout: Layout,
    pub base: i64,
    pub nls: Vec<Nl>,
    pub elided: Vec<usize>,
}

pub fn collapse_read(g: &GraphIr, read: &Read) -> Collapsed {
    let mut layout = read.layout.clone();
    let mut base = read.base;
    let mut src = read.src;
    let mut nls = Vec::new();
    let mut elided = Vec::new();
    if base == 0 {
        loop {
            let Some(pread) = is_view_or_pointwise(&g.nodes[src].kind) else { break };
            if pread.base != 0 {
                break;
            }
            let composed = match rowmajor_compose(&pread.layout, &layout) {
                Ok(l) => l,
                Err(_) => break,
            };
            if let NodeKind::Pointwise { nl, .. } = &g.nodes[src].kind {
                nls.push(*nl);
            }
            elided.push(src);
            layout = composed;
            src = pread.src;
        }
    }
    Collapsed { src, layout, base, nls, elided }
}

fn refine_read_views(g: &GraphIr, r: &Read) -> Read {
    let c = collapse_read(g, r);
    if c.nls.is_empty() && !c.elided.is_empty() {
        Read::with_nls(c.src, c.layout, c.base, Vec::new())
    } else {
        r.clone()
    }
}

fn refine_read_fused(g: &GraphIr, r: &Read) -> Read {
    let c = collapse_read(g, r);
    if c.elided.is_empty() {
        r.clone()
    } else {
        let mut nls = c.nls.clone();
        nls.reverse();
        Read::with_nls(c.src, c.layout, c.base, nls)
    }
}

fn refine_read_pointwise(g: &GraphIr, r: &Read) -> Read {
    let c = collapse_read(g, r);
    if c.nls.is_empty() {
        r.clone()
    } else {
        let mut nls = c.nls.clone();
        nls.reverse();
        Read::with_nls(c.src, c.layout, c.base, nls)
    }
}

fn refine_with(g: &GraphIr, rewrite: impl Fn(&GraphIr, &Read) -> Read, skip_contract: bool) -> GraphIr {
    let mut nodes = g.nodes.clone();
    for node in nodes.iter_mut() {
        match &mut node.kind {
            NodeKind::Move { a, b, kext, pair, .. } => {
                if skip_contract && b.is_some() && *pair && *kext > 1 {
                    continue;
                }
                *a = rewrite(g, a);
                if let Some(b) = b.as_mut() {
                    *b = rewrite(g, b);
                }
            }
            NodeKind::Pointwise { read, .. } => {
                *read = rewrite(g, read);
            }
            _ => {}
        }
    }
    GraphIr { nodes }
}

pub fn refine_views(g: &GraphIr) -> GraphIr {
    refine_with(g, refine_read_views, false)
}

pub fn refine_fused(g: &GraphIr) -> GraphIr {
    refine_with(g, refine_read_fused, false)
}

pub fn refine_pointwise(g: &GraphIr) -> GraphIr {
    refine_with(g, refine_read_pointwise, true)
}

fn remap_read(r: &Read, m: &HashMap<usize, usize>) -> Read {
    Read { src: m[&r.src], layout: r.layout.clone(), base: r.base, nls: r.nls.clone() }
}

fn remap_idx(ir: &IdxRead, m: &HashMap<usize, usize>) -> IdxRead {
    IdxRead { idx: m[&ir.idx], idx_layout: ir.idx_layout.clone(), stride: ir.stride }
}

fn remap_kind(k: &NodeKind, m: &HashMap<usize, usize>) -> NodeKind {
    match k {
        NodeKind::Input { .. } | NodeKind::LiteralF32(_) | NodeKind::LiteralI32(_) => k.clone(),
        NodeKind::Pointwise { nl, read } => NodeKind::Pointwise { nl: *nl, read: remap_read(read, m) },
        NodeKind::Move { a, b, kext, pair, a_idx, write_idx, write_base, reduce_max } => NodeKind::Move {
            a: remap_read(a, m),
            b: b.as_ref().map(|b| remap_read(b, m)),
            kext: *kext,
            pair: *pair,
            a_idx: a_idx.iter().map(|i| remap_idx(i, m)).collect(),
            write_idx: write_idx.iter().map(|i| remap_idx(i, m)).collect(),
            write_base: write_base.map(|w| m[&w]),
            reduce_max: *reduce_max,
        },
    }
}

pub fn prune(g: &GraphIr, roots: &[usize]) -> (GraphIr, Vec<usize>) {
    let mut keep = std::collections::HashSet::new();
    for &r in roots {
        for id in reachable(g, r) {
            keep.insert(id);
        }
    }
    let mut kept: Vec<usize> = keep.into_iter().collect();
    kept.sort_unstable();
    let m: HashMap<usize, usize> = kept.iter().enumerate().map(|(new, &old)| (old, new)).collect();
    let nodes: Vec<Node> = kept
        .iter()
        .map(|&old| Node {
            kind: remap_kind(&g.nodes[old].kind, &m),
            dims: g.nodes[old].dims.clone(),
            dtype: g.nodes[old].dtype,
        })
        .collect();
    let new_roots = roots.iter().map(|r| m[r]).collect();
    (GraphIr { nodes }, new_roots)
}

pub fn reachable(g: &GraphIr, root: usize) -> std::collections::HashSet<usize> {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match &g.nodes[id].kind {
            NodeKind::Move { a, b, a_idx, write_idx, write_base, .. } => {
                stack.push(a.src);
                if let Some(b) = b {
                    stack.push(b.src);
                }
                for ir in a_idx.iter().chain(write_idx.iter()) {
                    stack.push(ir.idx);
                }
                if let Some(wb) = write_base {
                    stack.push(*wb);
                }
            }
            NodeKind::Pointwise { read, .. } => stack.push(read.src),
            _ => {}
        }
    }
    seen
}

pub struct Frontier {
    pub elidable: Vec<usize>,
    pub boundary: Vec<(usize, usize, String)>,
}

pub fn fuse_frontier(g: &GraphIr) -> Frontier {
    let mut elidable = Vec::new();
    let mut boundary = Vec::new();
    for pid in 0..g.nodes.len() {
        let Some(pread) = is_view_or_pointwise(&g.nodes[pid].kind) else { continue };
        let mut all_ok = true;
        let mut any = false;
        for cid in 0..g.nodes.len() {
            for r in reads_of(&g.nodes[cid].kind) {
                if r.src == pid {
                    any = true;
                    if let Err(e) = edge_compose(pread, r) {
                        all_ok = false;
                        boundary.push((pid, cid, e));
                    }
                }
            }
        }
        if any && all_ok {
            elidable.push(pid);
        }
    }
    Frontier { elidable, boundary }
}

pub fn dump(g: &GraphIr) -> Vec<String> {
    let mut out = Vec::new();
    for (id, node) in g.nodes.iter().enumerate() {
        let reads: Vec<String> = match &node.kind {
            NodeKind::Pointwise { read, .. } => vec![format!("r{}@{}:{}", read.src, read.base, read.layout)],
            NodeKind::Move { a, b, .. } => {
                let mut v = vec![format!("a{}@{}:{}", a.src, a.base, a.layout)];
                if let Some(b) = b {
                    v.push(format!("b{}@{}:{}", b.src, b.base, b.layout));
                }
                v
            }
            _ => vec![],
        };
        out.push(format!("[{id}] {} dims={:?} {}", kind_tag(&node.kind), node.dims, reads.join("  ")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_model::{build, NativeCfg};

    fn refinement_inputs(g: &GraphIr, cfg: &NativeCfg) -> std::collections::HashMap<String, Vec<f32>> {
        let mut inputs = std::collections::HashMap::new();
        for node in &g.nodes {
            if let NodeKind::Input { name } = &node.kind {
                let n: i64 = node.dims.iter().product();
                let v: Vec<f32> = match name.as_str() {
                    "ids" => (0..n).map(|j| (j % cfg.vocab) as f32).collect(),
                    "scatter_idx" => (0..n).map(|j| (j % cfg.cap) as f32).collect(),
                    "mask" => crate::native_model::causal_mask(cfg),
                    _ => (0..n).map(|j| (((j * 7 + 1) % 17) as f32 - 8.0) * 0.05).collect(),
                };
                inputs.insert(name.clone(), v);
            }
        }
        inputs
    }

    #[test]
    fn refine_views_is_bit_exact_and_drops_dispatches() {
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 2, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let g = &nm.g;
        let inputs = refinement_inputs(g, &cfg);
        let fused = refine_views(g);
        let out = nm.logits.id;

        let m0 = crate::cpu_ref::eval(g, &inputs);
        let m1 = crate::cpu_ref::eval(&fused, &inputs);
        let (a, b) = (m0.f[out].as_ref().unwrap(), m1.f[out].as_ref().unwrap());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "logits differ after view refinement at {i}");
        }
        let (r0, r1) = (reachable(g, out), reachable(&fused, out));
        assert!(r1.len() < r0.len());
        println!("VIEW_REFINE orig={} refined={} dropped={}", r0.len(), r1.len(), r0.len() - r1.len());
    }

    #[test]
    fn refine_fused_is_bit_exact_and_drops_dispatches() {
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 2, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let g = &nm.g;
        let inputs = refinement_inputs(g, &cfg);
        let fused = refine_fused(g);
        let out = nm.logits.id;

        let m0 = crate::cpu_ref::eval(g, &inputs);
        let m1 = crate::cpu_ref::eval(&fused, &inputs);
        let (a, b) = (m0.f[out].as_ref().unwrap(), m1.f[out].as_ref().unwrap());
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "logits differ after pointwise+view fusion at {i}");
        }
        let (r0, r1) = (reachable(g, out), reachable(&fused, out));
        let dropped = r0.len() - r1.len();
        assert!(dropped >= 10, "pointwise+view fusion should elide many nodes, dropped {dropped}");
        println!("FUSED_REFINE orig={} refined={} dropped={}", r0.len(), r1.len(), dropped);
    }

    #[test]
    #[ignore]
    fn native_full_model_pointwise_fused() {
        use crate::native_exec::NativeExec;
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 2, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let g = &nm.g;
        let inputs = refinement_inputs(g, &cfg);

        let refm = crate::cpu_ref::eval(g, &inputs);
        let want = refm.f[nm.logits.id].as_ref().unwrap().clone();
        let close = |a: &[f32], b: &[f32]| -> f32 {
            let peak = b.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));
            a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / peak
        };

        unsafe { std::env::set_var("NOFUSE", "1") };
        let mut ex0 = NativeExec::new();
        ex0.build(g, &inputs);
        ex0.replay();
        let r0 = close(&ex0.read_f32(g, nm.logits.id), &want);
        let d0 = ex0.dispatches;

        unsafe { std::env::remove_var("NOFUSE") };
        let mut ex1 = NativeExec::new();
        ex1.build(g, &inputs);
        ex1.replay();
        let r1 = close(&ex1.read_f32(g, nm.logits.id), &want);
        let d1 = ex1.dispatches;

        println!("NATIVE_FULL raw_dispatches={d0} rel_raw={r0:.2e} | fused_dispatches={d1} rel_fused={r1:.2e}");
        assert!(r0 <= 1e-3, "raw native vs cpu_ref {r0:.2e}");
        assert!(r1 <= 1e-3, "pointwise-fused native vs cpu_ref {r1:.2e}");
        assert!(d1 < d0, "fused must dispatch fewer: {d1} vs {d0}");
    }

    #[test]
    #[ignore]
    fn native_inline_nls_matches_cpu_ref() {
        use crate::ir::Dt;
        use crate::native_exec::NativeExec;
        use std::collections::HashMap;

        let mut g = GraphIr::new();
        let x = g.input("x", vec![4, 8], Dt::F32);
        let w = g.input("w", vec![4, 8], Dt::F32);
        let e = g.pointwise(Nl::Exp2, x);
        let denom = g.sum_last(e);
        let rs = g.pointwise(Nl::Rsqrt, x);
        let prod = g.mul(w, rs);

        let mut inputs: HashMap<String, Vec<f32>> = HashMap::new();
        inputs.insert("x".into(), (0..32).map(|j| 0.5 + (j % 7) as f32 * 0.3).collect());
        inputs.insert("w".into(), (0..32).map(|j| ((j % 5) as f32 - 2.0) * 0.4).collect());

        let refm = crate::cpu_ref::eval(&g, &inputs);
        let (pruned, nr) = prune(&refine_fused(&g), &[denom.id, prod.id]);

        let mut ex = NativeExec::new();
        ex.build(&pruned, &inputs);
        ex.replay();

        let close = |a: &[f32], b: &[f32]| -> f32 {
            let peak = b.iter().fold(1e-30f32, |m, &x| m.max(x.abs()));
            a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / peak
        };

        let gd = ex.read_f32(&pruned, nr[0]);
        let gp = ex.read_f32(&pruned, nr[1]);
        let rd = close(&gd, refm.f[denom.id].as_ref().unwrap());
        let rp = close(&gp, refm.f[prod.id].as_ref().unwrap());
        println!("NATIVE_NLS dispatches={} rel_reduce_exp2={rd:.2e} rel_mul_rsqrt={rp:.2e}", ex.dispatches);
        assert!(rd <= 1e-4, "reduce+exp2 native vs cpu_ref {rd:.2e}");
        assert!(rp <= 1e-4, "mul+rsqrt native vs cpu_ref {rp:.2e}");
    }

    #[test]
    fn prune_after_fusion_is_bit_exact_and_smaller() {
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 2, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let g = &nm.g;
        let inputs = refinement_inputs(g, &cfg);

        let mut roots = vec![nm.logits.id];
        for (k, v) in &nm.cache_outs {
            roots.push(k.id);
            roots.push(v.id);
        }

        let fused = refine_fused(g);
        let (pruned, new_roots) = prune(&fused, &roots);

        let m0 = crate::cpu_ref::eval(g, &inputs);
        let m1 = crate::cpu_ref::eval(&pruned, &inputs);
        let a = m0.f[nm.logits.id].as_ref().unwrap();
        let b = m1.f[new_roots[0]].as_ref().unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "pruned+fused logits differ at {i}");
        }
        assert!(pruned.nodes.len() < g.nodes.len());
        println!("PRUNE nodes orig={} pruned={}", g.nodes.len(), pruned.nodes.len());
    }

    #[test]
    fn stable_softmax_cpu_ref_correct() {
        use crate::ir::Dt;
        use std::collections::HashMap;
        const LOG2E: f32 = 1.442_695;

        let mut g = GraphIr::new();
        let x = g.input("x", vec![2, 4], Dt::F32);
        let scaled = g.scale(x, LOG2E);
        let rowmax = g.max_last(scaled);
        let negmax = g.scale(rowmax, -1.0);
        let shifted = g.add_bcast_row2(scaled, negmax);
        let ex = g.pointwise(Nl::Exp2, shifted);
        let denom = g.sum_last(ex);
        let rd = g.pointwise(Nl::Recip, denom);
        let probs = g.mul_bcast_row2(ex, rd);

        let data = vec![1.0f32, 2.0, 3.0, 100.0, -5.0, 0.0, 5.0, 10.0];
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), data.clone());
        let m = crate::cpu_ref::eval(&g, &inputs);
        let p = m.f[probs.id].as_ref().unwrap();
        assert!(p.iter().all(|v| v.is_finite()), "stable softmax must not overflow: {p:?}");

        for r in 0..2 {
            let row = &data[r * 4..r * 4 + 4];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let s: f32 = exps.iter().sum();
            for c in 0..4 {
                let want = exps[c] / s;
                let got = p[r * 4 + c];
                assert!((got - want).abs() < 1e-5, "softmax[{r},{c}] got {got} want {want}");
            }
        }
    }

    #[test]
    fn rowmajor_compose_rejects_non_access_preserving() {
        use crate::ir::layout_of;
        let outer = layout_of(&[2, 3], &[1, 2]);
        let inner = layout_of(&[2, 2, 2], &[0, 2, 1]);
        let r = rowmajor_compose(&outer, &inner);
        assert!(r.is_err(), "non-nested outer must be rejected, got {r:?}");

        let a = layout_of(&[4, 4], &[4, 1]);
        let b = layout_of(&[4, 4], &[4, 1]);
        let ok = rowmajor_compose(&a, &b).expect("identity-ish compose must be accepted");
        for i in 0..ok.size() {
            assert_eq!(eval_at(&ok, i), eval_at(&a, eval_at(&b, i)));
        }
    }

    #[test]
    fn collapse_multi_pointwise_chain_bit_exact() {
        use crate::ir::{identity, Dt};
        use std::collections::HashMap;

        let mut g = GraphIr::new();
        let x = g.input("x", vec![8], Dt::F32);
        let e = g.pointwise(Nl::Exp2, x);
        let r = g.pointwise(Nl::Recip, e);

        let read = Read::new(r.id, identity(&[8]));
        let c = collapse_read(&g, &read);
        assert_eq!(c.src, x.id, "chain must collapse to the ultimate input");
        assert_eq!(c.elided, vec![r.id, e.id], "both pointwise nodes elided");
        assert_eq!(c.nls, vec![Nl::Recip, Nl::Exp2], "nls collected outer-first");

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![0.3f32, 0.7, 1.1, -0.4, 2.0, -1.3, 0.05, 1.7]);
        let m = crate::cpu_ref::eval(&g, &inputs);
        let apply_nl = |nl: Nl, v: f32| match nl {
            Nl::Exp2 => v.exp2(),
            Nl::Recip => 1.0 / v,
            _ => v,
        };
        for i in 0..8 {
            let orig = m.f[r.id].as_ref().unwrap()[eval_at(&read.layout, i) as usize];
            let mut v = m.f[c.src].as_ref().unwrap()[eval_at(&c.layout, i) as usize];
            for nl in c.nls.iter().rev() {
                v = apply_nl(*nl, v);
            }
            assert_eq!(v.to_bits(), orig.to_bits(), "multi-nl chain (rev order) mismatch at {i}");
        }
    }

    #[test]
    fn dump_real_graph_layouts() {
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 1, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        for line in dump(&nm.g) {
            println!("{line}");
        }
        let f = fuse_frontier(&nm.g);
        println!("ELIDABLE({})={:?}", f.elidable.len(), f.elidable);
        for (p, c, e) in &f.boundary {
            println!("BOUNDARY {p}->{c}: {e}");
        }
        println!("NODES={}", nm.g.nodes.len());
    }

    #[test]
    fn transitive_collapse_is_sound() {
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 1, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let g = &nm.g;
        let mut reads_checked = 0;
        let mut total_elided = 0;
        for cid in 0..g.nodes.len() {
            for r in reads_of(&g.nodes[cid].kind) {
                let c = collapse_read(g, r);

                assert_eq!(c.layout.size(), r.layout.size(), "collapse must preserve domain size at node {cid}");

                for &pid in &c.elided {
                    assert!(is_view_or_pointwise(&g.nodes[pid].kind).is_some(), "elided {pid} must be view/pointwise");
                }

                let producer_reads: Vec<&Read> = c.elided.iter()
                    .map(|&pid| is_view_or_pointwise(&g.nodes[pid].kind).unwrap())
                    .collect();

                let n = c.layout.size();
                for i in 0..n {
                    let mut off = eval_at(&r.layout, i);
                    for pr in &producer_reads {
                        off = eval_at(&pr.layout, off);
                    }
                    let got = eval_at(&c.layout, i);
                    assert_eq!(got, off, "collapsed layout must equal the chained access at node {cid} coord {i}");
                }

                total_elided += c.elided.len();
                reads_checked += 1;
            }
        }
        assert!(reads_checked > 0);
        assert!(total_elided > 0, "expected the graph to admit transitive elisions");
        println!("READS_CHECKED={reads_checked} TOTAL_ELIDED={total_elided}");
    }

    #[test]
    fn transitive_collapse_reproduces_values_bit_exact() {
        use crate::ir::Dt;
        use std::collections::HashMap;

        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 1, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let g = &nm.g;

        let mut inputs: HashMap<String, Vec<f32>> = HashMap::new();
        for node in &g.nodes {
            if let NodeKind::Input { name } = &node.kind {
                let n: i64 = node.dims.iter().product();
                let v: Vec<f32> = match name.as_str() {
                    "ids" => (0..n).map(|j| (j % cfg.vocab) as f32).collect(),
                    "scatter_idx" => (0..n).map(|j| (j % cfg.cap) as f32).collect(),
                    "mask" => crate::native_model::causal_mask(&cfg),
                    _ => (0..n).map(|j| (((j * 7 + 1) % 17) as f32 - 8.0) * 0.05).collect(),
                };
                inputs.insert(name.clone(), v);
            }
        }

        let apply_nl = |nl: Nl, x: f32| -> f32 {
            match nl {
                Nl::Exp2 => x.exp2(),
                Nl::Recip => 1.0 / x,
                Nl::Sqrt => x.sqrt(),
                Nl::Rsqrt => 1.0 / x.sqrt(),
                Nl::Relu => x.max(0.0),
                Nl::Step => if x > 0.0 { 1.0 } else { 0.0 },
                Nl::CastF32 => x,
            }
        };

        let m = crate::cpu_ref::eval(g, &inputs);
        let mut elements_checked = 0usize;
        for cid in 0..g.nodes.len() {
            for r in reads_of(&g.nodes[cid].kind) {
                let c = collapse_read(g, r);
                if c.elided.is_empty() {
                    continue;
                }
                let prod = m.f[r.src].as_ref().expect("elidable producer is f32");
                for i in 0..r.layout.size() {
                    let orig = prod[eval_at(&r.layout, i) as usize];
                    let addr = eval_at(&c.layout, i) as usize;
                    let mut v = match g.nodes[c.src].dtype {
                        Dt::F32 => m.f[c.src].as_ref().unwrap()[addr],
                        Dt::I32 => m.i[c.src].as_ref().unwrap()[addr] as f32,
                    };
                    for nl in c.nls.iter().rev() {
                        v = apply_nl(*nl, v);
                    }
                    assert_eq!(
                        v.to_bits(), orig.to_bits(),
                        "collapsed value != producer value at node {cid} read-src {} coord {i}", r.src
                    );
                    elements_checked += 1;
                }
            }
        }
        assert!(elements_checked > 0, "expected elidable reads with values to check");
        println!("ELEMENTS_CHECKED={elements_checked}");
    }

    #[test]
    fn composition_elides_definitionally() {
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 1, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let g = &nm.g;
        let mut checked = 0;
        for pid in 0..g.nodes.len() {
            let Some(pread) = is_view_or_pointwise(&g.nodes[pid].kind) else { continue };
            for cid in 0..g.nodes.len() {
                for r in reads_of(&g.nodes[cid].kind) {
                    if r.src != pid {
                        continue;
                    }
                    if let Ok(fused) = edge_compose(pread, r) {
                        let n = fused.layout.size().min(64);
                        for i in 0..n {
                            let via = eval_at(&pread.layout, eval_at(&r.layout, i));
                            let got = eval_at(&fused.layout, i);
                            assert_eq!(got, via, "fused({i}) must equal producer(consumer({i})) for edge {pid}->{cid}");
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "expected at least one fusible view/pointwise edge");
        println!("DEFINITIONAL_EDGES_CHECKED={checked}");
    }
}
