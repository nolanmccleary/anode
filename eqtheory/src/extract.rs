use crate::coarsen::{consumers, fusion_regions, reduce_regions, region_root, ReduceRegion};
use crate::compose::reads_of;
use crate::ir::{GraphIr, NodeKind};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub enum Servicer {
    Primitive,
    Contraction,
    Elementwise,
    Reduce(ReduceRegion),
    Flash { scores: usize, pv: usize, mask: usize },
}

fn is_leaf(g: &GraphIr, id: usize) -> bool {
    matches!(
        g.nodes[id].kind,
        NodeKind::Input { .. } | NodeKind::LiteralF32(_) | NodeKind::LiteralI32(_)
    )
}

pub fn flash_cone(g: &GraphIr, out: usize, rroot: &HashMap<usize, ReduceRegion>) -> Option<(Vec<usize>, usize, usize)> {
    let probs_src = match &g.nodes[out].kind {
        NodeKind::Move { a, b: Some(_), pair: true, kext, a_idx, write_idx, write_base, .. }
            if *kext > 1 && a_idx.is_empty() && write_idx.is_empty() && write_base.is_none() =>
        {
            a.src
        }
        _ => return None,
    };
    let rr = rroot.get(&probs_src)?;
    let first_reduce = rr.stages[0].reduce;
    let red_in = match &g.nodes[first_reduce].kind {
        NodeKind::Move { a, .. } => a.src,
        _ => return None,
    };
    let mut cur = red_in;
    let mut chain = Vec::new();
    let mut guard = 0;
    let sc = loop {
        guard += 1;
        if guard > 64 {
            return None;
        }
        if is_contraction(&g.nodes[cur].kind) {
            break cur;
        }
        chain.push(cur);
        match &g.nodes[cur].kind {
            NodeKind::Pointwise { read, .. } => cur = read.src,
            NodeKind::Move { a, b, kext, a_idx, write_idx, write_base, .. }
                if *kext <= 2 && a_idx.is_empty() && write_idx.is_empty() && write_base.is_none() =>
            {
                if !is_leaf(g, a.src) {
                    cur = a.src;
                } else if let Some(b) = b {
                    if !is_leaf(g, b.src) {
                        cur = b.src;
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    };
    let mut mask = None;
    for &c in &chain {
        if let NodeKind::Move { a, b: Some(b), pair: false, kext: 2, .. } = &g.nodes[c].kind {
            for cand in [a.src, b.src] {
                if matches!(g.nodes[cand].kind, NodeKind::Input { .. }) && g.nodes[cand].dims.len() == 2 {
                    mask = Some(cand);
                }
            }
        }
    }
    let mask = mask?;
    let mut cone = rr.nodes();
    cone.extend(chain);
    cone.push(out);
    cone.push(sc);
    cone.sort_unstable();
    cone.dedup();
    Some((cone, sc, mask))
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub nodes: Vec<usize>,
    pub out: usize,
    pub servicer: Servicer,
}

fn is_contraction(k: &NodeKind) -> bool {
    matches!(k, NodeKind::Move { b: Some(_), pair: true, kext, a_idx, write_idx, write_base, .. }
        if *kext > 1 && a_idx.is_empty() && write_idx.is_empty() && write_base.is_none())
}

fn input_edges(g: &GraphIr, id: usize) -> Vec<usize> {
    let mut v: Vec<usize> = reads_of(&g.nodes[id].kind).iter().map(|r| r.src).collect();
    if let NodeKind::Move { a_idx, write_idx, write_base, .. } = &g.nodes[id].kind {
        for ir in a_idx.iter().chain(write_idx.iter()) {
            v.push(ir.idx);
        }
        if let Some(wb) = write_base {
            v.push(*wb);
        }
    }
    v
}

pub fn partition(g: &GraphIr, outputs: &[usize]) -> Vec<Kernel> {
    let cons = consumers(g);
    let rroot: HashMap<usize, ReduceRegion> =
        reduce_regions(g).into_iter().map(|r| (r.root(), r)).collect();
    let eroot: HashMap<usize, Vec<usize>> = fusion_regions(g)
        .into_iter()
        .map(|reg| (region_root(&reg, &cons), reg))
        .collect();

    let mut kernels = Vec::new();
    let mut assigned: HashSet<usize> = HashSet::new();
    let mut work: Vec<usize> = outputs.to_vec();
    while let Some(out) = work.pop() {
        if assigned.contains(&out) {
            continue;
        }
        if matches!(
            &g.nodes[out].kind,
            NodeKind::Input { .. } | NodeKind::LiteralF32(_) | NodeKind::LiteralI32(_)
        ) {
            assigned.insert(out);
            continue;
        }
        let (cone, servicer) = if let Some((cone, sc, mask)) = flash_cone(g, out, &rroot) {
            (cone, Servicer::Flash { scores: sc, pv: out, mask })
        } else if let Some(rr) = rroot.get(&out) {
            (rr.nodes(), Servicer::Reduce(rr.clone()))
        } else if let Some(reg) = eroot.get(&out) {
            (reg.clone(), Servicer::Elementwise)
        } else if is_contraction(&g.nodes[out].kind) {
            (vec![out], Servicer::Contraction)
        } else {
            (vec![out], Servicer::Primitive)
        };
        let coneset: HashSet<usize> = cone.iter().copied().collect();
        for &n in &cone {
            assigned.insert(n);
            for src in input_edges(g, n) {
                if !coneset.contains(&src) {
                    work.push(src);
                }
            }
        }
        kernels.push(Kernel { nodes: cone, out, servicer });
    }
    kernels.sort_by_key(|k| k.out);
    kernels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_model::{build, NativeCfg};

    #[test]
    fn partition_covers_compute_nodes() {
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 1, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let mut outs = vec![nm.logits.id];
        for (k, v) in &nm.cache_outs {
            outs.push(k.id);
            outs.push(v.id);
        }
        let ks = partition(&nm.g, &outs);
        let mut covered: HashSet<usize> = HashSet::new();
        for k in &ks {
            for &n in &k.nodes {
                assert!(covered.insert(n), "node {n} in two kernels");
            }
        }
        let mut n_prim = 0;
        let mut n_ew = 0;
        let mut n_red = 0;
        let mut n_con = 0;
        let mut n_flash = 0;
        for k in &ks {
            match k.servicer {
                Servicer::Primitive => n_prim += 1,
                Servicer::Elementwise => n_ew += 1,
                Servicer::Reduce(_) => n_red += 1,
                Servicer::Contraction => n_con += 1,
                Servicer::Flash { scores, pv, mask } => {
                    n_flash += 1;
                    println!("  FLASH cone={} nodes (scores={scores} pv={pv} mask={mask})", k.nodes.len());
                }
            }
        }
        assert!(n_flash >= 1, "flash servicer should claim the attention composition");
        for (id, node) in nm.g.nodes.iter().enumerate() {
            if !matches!(node.kind, NodeKind::Input { .. } | NodeKind::LiteralF32(_) | NodeKind::LiteralI32(_)) {
                assert!(covered.contains(&id), "compute node {id} uncovered");
            }
        }
        println!("KERNELS={} prim={n_prim} ew={n_ew} reduce={n_red} contract={n_con} flash={n_flash}", ks.len());
    }
}
