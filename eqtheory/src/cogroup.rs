use crate::coarsen::consumers;
use crate::compose::reads_of;
use crate::ir::{layout_of, Dt, GraphIr, Node, NodeKind, Read};
use std::collections::{BTreeMap, HashSet};

fn leaves(t: &cute_layout::IntTuple) -> Vec<i64> {
    match t {
        cute_layout::IntTuple::Leaf(v) => vec![*v],
        cute_layout::IntTuple::Tuple(ts) => ts.iter().flat_map(leaves).collect(),
    }
}

fn axes_of(l: &cute_layout::Layout) -> (Vec<i64>, Vec<i64>) {
    (leaves(&l.shape), leaves(&l.stride))
}

#[derive(Debug, Clone)]
pub struct ConcatRecipe {
    pub out_name: String,
    pub comps: Vec<String>,
    pub n_each: Vec<i64>,
    pub bn: Vec<i64>,
    pub bk: Vec<i64>,
    pub k: i64,
    pub n_total: i64,
}

fn proj(g: &GraphIr, id: usize) -> Option<(usize, usize, i64, i64, i64)> {
    if let NodeKind::Move { a, b: Some(b), kext, pair: true, a_idx, write_idx, write_base, reduce_max: false } = &g.nodes[id].kind {
        if *kext <= 1 || !a_idx.is_empty() || !write_idx.is_empty() || write_base.is_some() {
            return None;
        }
        let dims = &g.nodes[id].dims;
        if dims.len() != 2 {
            return None;
        }
        if !matches!(g.nodes[b.src].kind, NodeKind::Input { .. }) {
            return None;
        }
        if !a.nls.is_empty() || !b.nls.is_empty() || a.base != 0 || b.base != 0 {
            return None;
        }
        return Some((a.src, b.src, *kext, dims[0], dims[1]));
    }
    None
}

fn remap_reads(kind: &mut NodeKind, target: usize, mut f: impl FnMut(&mut Read)) {
    match kind {
        NodeKind::Pointwise { read, .. } => {
            if read.src == target {
                f(read);
            }
        }
        NodeKind::Move { a, b, .. } => {
            if a.src == target {
                f(a);
            }
            if let Some(b) = b {
                if b.src == target {
                    f(b);
                }
            }
        }
        _ => {}
    }
}

pub fn coarsen_cogroup(g: &mut GraphIr) -> (Vec<ConcatRecipe>, HashSet<usize>) {
    let n0 = g.nodes.len();
    let cons = consumers(g);
    let mut groups: BTreeMap<(usize, i64, i64), Vec<usize>> = BTreeMap::new();
    for id in 0..n0 {
        if let Some((a_src, _w, kext, m, _n)) = proj(g, id) {
            groups.entry((a_src, kext, m)).or_default().push(id);
        }
    }
    let mut recipes = Vec::new();
    let mut dead: HashSet<usize> = HashSet::new();
    let mut gi = 0;
    for ((a_src, kext, m), members) in groups {
        if members.len() < 2 {
            continue;
        }
        let mut comps = Vec::new();
        let mut weights = Vec::new();
        let mut bn = Vec::new();
        let mut bk = Vec::new();
        let mut ok = true;
        for &c in &members {
            let (w, bst) = match &g.nodes[c].kind {
                NodeKind::Move { b: Some(b), .. } => (b.src, axes_of(&b.layout).1),
                _ => {
                    ok = false;
                    break;
                }
            };
            let name = match &g.nodes[w].kind {
                NodeKind::Input { name } => name.clone(),
                _ => {
                    ok = false;
                    break;
                }
            };
            if cons[c].iter().any(|&cid| !reads_of(&g.nodes[cid].kind).iter().any(|r| r.src == c)) {
                ok = false;
                break;
            }
            weights.push(w);
            comps.push(name);
            bn.push(bst[bst.len() - 2]);
            bk.push(bst[bst.len() - 1]);
        }
        if !ok {
            continue;
        }

        let n_each: Vec<i64> = members.iter().map(|&c| g.nodes[c].dims[1]).collect();
        let n_total: i64 = n_each.iter().sum();
        let c0 = members[0];
        let qkv = *members.last().unwrap();

        if members.iter().any(|&c| cons[c].iter().any(|&cid| cid <= qkv)) {
            continue;
        }

        let a0 = match &g.nodes[c0].kind {
            NodeKind::Move { a, .. } => a.clone(),
            _ => unreachable!(),
        };
        let (sha, sta) = axes_of(&a0.layout);
        let out_name = format!("__cogroup_{gi}_{a_src}");
        gi += 1;

        g.nodes[c0] = Node {
            kind: NodeKind::Input { name: out_name.clone() },
            dims: vec![kext, n_total],
            dtype: Dt::F32,
        };

        let a_wide = Read {
            src: a_src,
            layout: layout_of(&[sha[0], n_total, sha[2]], &[sta[0], 0, sta[2]]),
            base: 0,
            nls: vec![],
        };
        let b_wide = Read {
            src: c0,
            layout: layout_of(&[m, n_total, kext], &[0, 1, n_total]),
            base: 0,
            nls: vec![],
        };
        g.nodes[qkv] = Node {
            kind: NodeKind::Move {
                a: a_wide,
                b: Some(b_wide),
                kext,
                pair: true,
                a_idx: vec![],
                write_idx: vec![],
                write_base: None,
                reduce_max: false,
            },
            dims: vec![m, n_total],
            dtype: Dt::F32,
        };

        let mut off = 0i64;
        for (mi, &c) in members.iter().enumerate() {
            let n_x = n_each[mi];
            for &cid in &cons[c] {
                remap_reads(&mut g.nodes[cid].kind, c, |r| {
                    let (sh, st) = axes_of(&r.layout);
                    let new_st: Vec<i64> = st
                        .iter()
                        .map(|&s| if s > 0 && s % n_x == 0 { s / n_x * n_total } else { s })
                        .collect();
                    r.src = qkv;
                    r.layout = layout_of(&sh, &new_st);
                    r.base = (r.base / n_x) * n_total + off + (r.base % n_x);
                });
            }
            if c != c0 && c != qkv {
                dead.insert(c);
            }
            off += n_x;
        }
        for &w in &weights {
            if cons[w].iter().all(|&cid| members.contains(&cid)) {
                dead.insert(w);
            }
        }
        recipes.push(ConcatRecipe { out_name, comps, n_each, bn, bk, k: kext, n_total });
    }
    (recipes, dead)
}
