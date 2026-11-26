use crate::compose::reads_of;
use crate::ir::{rm_strides, GraphIr, NodeKind, Read};

fn leaves(t: &cute_layout::IntTuple) -> Vec<i64> {
    match t {
        cute_layout::IntTuple::Leaf(v) => vec![*v],
        cute_layout::IntTuple::Tuple(ts) => ts.iter().flat_map(leaves).collect(),
    }
}

pub fn contig_edge(r: &Read, dims: &[i64]) -> bool {
    let sh = leaves(&r.layout.shape);
    let st = leaves(&r.layout.stride);
    if sh.len() < dims.len() {
        return false;
    }
    let rm = rm_strides(dims);
    sh[..dims.len()] == *dims && st[..dims.len()] == rm[..] && r.base == 0 && r.nls.is_empty()
}

fn read_from(k: &NodeKind, src: usize) -> Option<&Read> {
    reads_of(k).into_iter().find(|r| r.src == src)
}

pub fn is_elementwise(k: &NodeKind) -> bool {
    match k {
        NodeKind::Pointwise { .. } => true,
        NodeKind::Move { a: _, b, kext, pair, a_idx, write_idx, write_base, reduce_max } => {
            let contraction = b.is_some() && *pair && *kext > 1;
            let reduction = b.is_none() && *kext > 1;
            !contraction
                && !reduction
                && !*reduce_max
                && a_idx.is_empty()
                && write_idx.is_empty()
                && write_base.is_none()
        }
        _ => false,
    }
}

pub fn consumers(g: &GraphIr) -> Vec<Vec<usize>> {
    let mut cons = vec![Vec::new(); g.nodes.len()];
    for (cid, node) in g.nodes.iter().enumerate() {
        for r in reads_of(&node.kind) {
            cons[r.src].push(cid);
        }
        if let NodeKind::Move { a_idx, write_idx, write_base, .. } = &node.kind {
            for ir in a_idx.iter().chain(write_idx.iter()) {
                cons[ir.idx].push(cid);
            }
            if let Some(wb) = write_base {
                cons[*wb].push(cid);
            }
        }
    }
    cons
}

pub fn region_root(region: &[usize], cons: &[Vec<usize>]) -> usize {
    let inset: std::collections::HashSet<usize> = region.iter().copied().collect();
    for &id in region {
        if cons[id].iter().all(|c| !inset.contains(c)) {
            return id;
        }
    }
    *region.last().unwrap()
}

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while parent[r] != r {
        r = parent[r];
    }
    let mut c = x;
    while parent[c] != r {
        let n = parent[c];
        parent[c] = r;
        c = n;
    }
    r
}

pub fn fusion_regions(g: &GraphIr) -> Vec<Vec<usize>> {
    let n = g.nodes.len();
    let cons = consumers(g);
    let mut parent: Vec<usize> = (0..n).collect();
    for pid in 0..n {
        if !is_elementwise(&g.nodes[pid].kind) {
            continue;
        }
        if cons[pid].len() != 1 {
            continue;
        }
        let cid = cons[pid][0];
        if !is_elementwise(&g.nodes[cid].kind) {
            continue;
        }
        if g.nodes[pid].dims != g.nodes[cid].dims {
            continue;
        }
        let Some(edge) = read_from(&g.nodes[cid].kind, pid) else { continue };
        if !contig_edge(edge, &g.nodes[cid].dims) {
            continue;
        }
        let (ra, rb) = (find(&mut parent, pid), find(&mut parent, cid));
        parent[ra.min(rb)] = ra.max(rb);
    }
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for id in 0..n {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push(id);
    }
    let mut regions: Vec<Vec<usize>> = groups.into_values().filter(|v| v.len() >= 2).collect();
    for r in regions.iter_mut() {
        r.sort_unstable();
    }
    regions.sort_by_key(|r| r[0]);
    regions
}

#[derive(Debug, Clone)]
pub struct Stage {
    pub reduce: usize,
    pub reduce_max: bool,
    pub post: Vec<usize>,
    pub bcast: usize,
}

#[derive(Debug, Clone)]
pub struct ReduceRegion {
    pub stages: Vec<Stage>,
    pub pre0: Vec<usize>,
    pub inter: Vec<usize>,
}

impl ReduceRegion {
    pub fn root(&self) -> usize {
        self.stages.last().unwrap().bcast
    }
    pub fn nodes(&self) -> Vec<usize> {
        let mut v = self.pre0.clone();
        v.extend(&self.inter);
        for s in &self.stages {
            v.push(s.reduce);
            v.extend(&s.post);
            v.push(s.bcast);
        }
        v
    }
    pub fn internal(&self) -> std::collections::HashSet<usize> {
        let mut s: std::collections::HashSet<usize> = self.pre0.iter().copied().collect();
        s.extend(&self.inter);
        for st in &self.stages {
            s.insert(st.bcast);
        }
        s
    }
}

fn numel(d: &[i64]) -> i64 {
    d.iter().product()
}

fn ewise_cone(g: &GraphIr, start: usize, cons: &[Vec<usize>], acc: &mut Vec<usize>) {
    if !is_elementwise(&g.nodes[start].kind) || cons[start].len() != 1 {
        return;
    }
    if acc.contains(&start) {
        return;
    }
    acc.push(start);
    for r in reads_of(&g.nodes[start].kind) {
        ewise_cone(g, r.src, cons, acc);
    }
}

fn is_reduction(k: &NodeKind) -> Option<bool> {
    if let NodeKind::Move { b: None, kext, reduce_max, a_idx, write_idx, write_base, .. } = k {
        if *kext > 1 && a_idx.is_empty() && write_idx.is_empty() && write_base.is_none() {
            return Some(*reduce_max);
        }
    }
    None
}

fn find_post_bcast(g: &GraphIr, cons: &[Vec<usize>], reduce: usize) -> Option<(Vec<usize>, usize)> {
    let mut post = Vec::new();
    let mut cur = reduce;
    loop {
        if cons[cur].len() != 1 {
            return None;
        }
        let c = cons[cur][0];
        let is_combine = matches!(&g.nodes[c].kind, NodeKind::Move { b: Some(_), .. })
            && read_from(&g.nodes[c].kind, cur).is_some()
            && numel(&g.nodes[c].dims) > numel(&g.nodes[cur].dims);
        if is_combine {
            return Some((post, c));
        }
        if is_elementwise(&g.nodes[c].kind) && g.nodes[c].dims == g.nodes[cur].dims {
            post.push(c);
            cur = c;
        } else {
            return None;
        }
    }
}

fn find_next_reduce(g: &GraphIr, cons: &[Vec<usize>], bcast: usize, kext: i64) -> Option<(Vec<usize>, usize)> {
    let mut inter = Vec::new();
    let mut stack = vec![bcast];
    let mut seen = std::collections::HashSet::new();
    while let Some(cur) = stack.pop() {
        for &c in &cons[cur] {
            if !seen.insert(c) {
                continue;
            }
            if let Some(_) = is_reduction(&g.nodes[c].kind) {
                if matches!(&g.nodes[c].kind, NodeKind::Move { kext: k, .. } if *k == kext) {
                    return Some((inter, c));
                }
            } else if is_elementwise(&g.nodes[c].kind) {
                inter.push(c);
                stack.push(c);
            }
        }
    }
    None
}

pub fn reduce_regions(g: &GraphIr) -> Vec<ReduceRegion> {
    let cons = consumers(g);
    let mut out = Vec::new();
    let mut used_reduce: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for rid in 0..g.nodes.len() {
        if used_reduce.contains(&rid) {
            continue;
        }
        let Some(_) = is_reduction(&g.nodes[rid].kind) else { continue };
        let NodeKind::Move { a, kext, .. } = &g.nodes[rid].kind else { continue };
        let kext = *kext;
        let mut pre0 = Vec::new();
        ewise_cone(g, a.src, &cons, &mut pre0);

        let mut stages: Vec<Stage> = Vec::new();
        let mut inter: Vec<usize> = Vec::new();
        let mut cur = rid;
        let ok = loop {
            let reduce_max = is_reduction(&g.nodes[cur].kind).unwrap();
            let Some((post, bcast)) = find_post_bcast(g, &cons, cur) else { break false };
            stages.push(Stage { reduce: cur, reduce_max, post, bcast });
            match find_next_reduce(g, &cons, bcast, kext) {
                Some((mid, next)) => {
                    inter.extend(mid);
                    cur = next;
                }
                None => break true,
            }
        };
        if !ok || stages.is_empty() {
            continue;
        }
        let region = ReduceRegion { stages, pre0, inter };
        let nodes = region.nodes();
        if nodes.iter().any(|id| used.contains(id)) {
            continue;
        }
        for id in &nodes {
            used.insert(*id);
        }
        for s in &region.stages {
            used_reduce.insert(s.reduce);
        }
        out.push(region);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_model::{build, NativeCfg};

    #[test]
    fn report_fusion_regions() {
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 1, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let regions = fusion_regions(&nm.g);
        let total: usize = regions.iter().map(|r| r.len()).sum();
        println!("REGIONS={} nodes_fused={} total_nodes={}", regions.len(), total, nm.g.nodes.len());
        let tag = |id: usize| -> String {
            match &nm.g.nodes[id].kind {
                NodeKind::Pointwise { nl, .. } => format!("{nl:?}"),
                NodeKind::Move { b, kext, pair, .. } => {
                    if b.is_some() && *pair {
                        format!("mul(k{kext})")
                    } else if b.is_some() {
                        format!("add(k{kext})")
                    } else {
                        "view".into()
                    }
                }
                _ => "?".into(),
            }
        };
        for r in &regions {
            let kinds: Vec<String> = r.iter().map(|&id| tag(id)).collect();
            println!("  region {:?}: {}", r, kinds.join(" + "));
        }
        let rr = reduce_regions(&nm.g);
        println!("REDUCE_REGIONS={}", rr.len());
        for r in &rr {
            let st: Vec<String> = r.stages.iter().map(|s| format!("r{}({}){:?}->b{}", s.reduce, if s.reduce_max {"max"} else {"sum"}, s.post, s.bcast)).collect();
            println!("  reduce-region pre0={:?} inter={:?} stages={}", r.pre0, r.inter, st.join(" => "));
        }
    }
}
