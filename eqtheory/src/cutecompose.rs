use cute_layout::algebra::mapping::{coalesce, composition};
use cute_layout::{IntTuple, Layout};
use eqtheory::cute_probe::probe;
use eqtheory::loaders::{self, ModelConfig};
use luminal::logical_ir::{LogicalOp, MapEntry};
use luminal::prelude::petgraph::Direction;
use luminal::prelude::NodeIndex;

fn cfg() -> ModelConfig {
    ModelConfig {
        vocab: 32_000,
        d: 1024,
        ff: 2752,
        n_heads: 16,
        n_kv_heads: 4,
        layers: 2,
        cache_capacity: 4,
    }
}

fn dims_of(g: &luminal::prelude::petgraph::stable_graph::StableDiGraph<luminal::logical_ir::LogicalNode, luminal::logical_ir::InputPort>, id: NodeIndex) -> Option<Vec<i64>> {
    g[id].dims.iter().map(|d| Some(d.to_usize()? as i64)).collect()
}

fn rm_strides(dims: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        s[k] = s[k + 1] * dims[k + 1];
    }
    s
}

fn colex_strides(dims: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; dims.len()];
    for k in 1..dims.len() {
        s[k] = s[k - 1] * dims[k - 1];
    }
    s
}

fn eval_entry(e: &MapEntry, coords: &[i64]) -> Option<i64> {
    eqtheory::cute_probe::eval_entry(e, coords)
}

fn fused_offset(chain: &[(Vec<MapEntry>, Vec<i64>)], base_strides: &[i64], coord: &[i64]) -> Option<i64> {
    let mut c = coord.to_vec();
    for (entries, _) in chain.iter().rev().take(chain.len() - 1) {
        let next: Option<Vec<i64>> = entries.iter().map(|e| eval_entry(e, &c)).collect();
        c = next?;
    }
    let (first_entries, _) = &chain[0];
    let mut acc = 0i64;
    for (e, s) in first_entries.iter().zip(base_strides) {
        acc += eval_entry(e, &c)? * s;
    }
    Some(acc)
}

fn colex(i: i64, dims: &[i64]) -> Vec<i64> {
    let mut c = vec![0i64; dims.len()];
    let mut r = i;
    for (k, d) in dims.iter().enumerate() {
        c[k] = r % d;
        r /= d;
    }
    c
}

fn functional_eq(a: &Layout, b: &Layout, n: i64) -> bool {
    (0..n).all(|i| a.checked_call(&IntTuple::Leaf(i)) == b.checked_call(&IntTuple::Leaf(i)))
}

fn main() {
    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load(&cfg());
    luminal::logical_ir::set_cute_lowering(false);
    let g = loaded.graph.logical.petgraph();

    let is_view = |id: NodeIndex| matches!(g[id].op, LogicalOp::IndexMapApply { .. });
    let mut chains: Vec<Vec<NodeIndex>> = Vec::new();
    for id in g.node_indices() {
        if !is_view(id) {
            continue;
        }
        let terminal = !g
            .neighbors_directed(id, Direction::Outgoing)
            .any(|c| is_view(c));
        if !terminal {
            continue;
        }
        let mut chain = vec![id];
        let mut cur = id;
        loop {
            let src = g.neighbors_directed(cur, Direction::Incoming).next().unwrap();
            if is_view(src) {
                chain.push(src);
                cur = src;
            } else {
                break;
            }
        }
        if chain.len() >= 2 {
            chain.reverse();
            chains.push(chain);
        }
    }

    println!("INTERIOR VIEW CHAINS composed by cute composition (eq 16/17) + eq 18 check.");
    println!("per chain: probe+verify each link, compose via the API, verify the composite");
    println!("against the fused MapEntry evaluation, then compare both associations.\n");
    println!("{} chains of length >=2 in the loaded model\n", chains.len());

    let (mut ok_both, mut assoc_holds, mut precond_holds) = (0usize, 0usize, 0usize);
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for chain in &chains {
        let base = g.neighbors_directed(chain[0], Direction::Incoming).next().unwrap();
        let base_dims = dims_of(g, base).expect("concrete base");
        let base_strides = rm_strides(&base_dims);

        let mut layouts: Vec<Layout> = Vec::new();
        let mut entry_chain: Vec<(Vec<MapEntry>, Vec<i64>)> = Vec::new();
        let mut probe_ok = true;
        for (k, &v) in chain.iter().enumerate() {
            let src = g.neighbors_directed(v, Direction::Incoming).next().unwrap();
            let src_dims = dims_of(g, src).expect("concrete");
            let out_dims = dims_of(g, v).expect("concrete");
            let strides = colex_strides(&src_dims);
            let LogicalOp::IndexMapApply { entries } = &g[v].op else { unreachable!() };
            match probe(entries, &out_dims, &strides) {
                Some(p) if p.offset == 0 => {
                    layouts.push(p.layout.parse().unwrap());
                    entry_chain.push((entries.clone(), out_dims));
                }
                _ => {
                    probe_ok = false;
                    break;
                }
            }
        }
        if !probe_ok {
            println!("  chain via {:?}: link not probe-representable, skipped", chain);
            continue;
        }
        let sh: Vec<String> = base_dims.iter().map(|d| d.to_string()).collect();
        let st: Vec<String> = base_strides.iter().map(|d| d.to_string()).collect();
        let a_base: Layout = format!("({}):({})", sh.join(","), st.join(",")).parse().unwrap();
        layouts.insert(0, a_base);

        let left = layouts[1..]
            .iter()
            .try_fold(layouts[0].clone(), |acc, l| composition(&acc, l));
        let right = layouts[..layouts.len() - 1]
            .iter()
            .rev()
            .try_fold(layouts[layouts.len() - 1].clone(), |acc, l| {
                composition(l, &acc)
            });

        let term_dims = &entry_chain.last().unwrap().1;
        let n: i64 = term_dims.iter().product();
        let key = layouts
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(" o ");
        if !seen.insert(key.clone()) {
            continue;
        }

        let precond = layouts
            .windows(2)
            .all(|w| w[1].cosize() <= w[0].size());
        if precond {
            precond_holds += 1;
        }

        match (left, right) {
            (Ok(lc), Ok(rc)) => {
                let lc = coalesce(&lc);
                let rc = coalesce(&rc);
                let check_at = |i: i64| {
                    let c = colex(i, term_dims);
                    lc.checked_call(&IntTuple::Leaf(i)) == fused_offset(&entry_chain, &base_strides, &c)
                };
                let fused_ok = if n <= 1 << 20 {
                    (0..n).all(check_at)
                } else {
                    let mut st = 0x9e3779b97f4a7c15u64;
                    (0..65536).all(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        check_at(((st >> 33) as i64).rem_euclid(n))
                    })
                };
                let assoc = functional_eq(&lc, &rc, n.min(1 << 20));
                if fused_ok {
                    ok_both += 1;
                }
                if assoc {
                    assoc_holds += 1;
                }
                println!(
                    "  terms {}  {}\n         -> {}\n         fused-map verified over {n}: {}   eq18 precond(image(B) in Z(A)): {}   both associations equal: {}",
                    layouts.len(),
                    key,
                    lc,
                    if fused_ok { "YES" } else { "NO" },
                    if precond { "holds" } else { "VIOLATED" },
                    if assoc { "YES" } else { "NO" }
                );
            }
            (l, r) => {
                println!(
                    "  terms {}  {}  -> composition REFUSED (left: {}, right: {}) -> chain materializes",
                    layouts.len(),
                    key,
                    l.err().map(|e| e.to_string()).unwrap_or_else(|| "ok".into()),
                    r.err().map(|e| e.to_string()).unwrap_or_else(|| "ok".into())
                );
            }
        }
    }
    println!(
        "\nsummary: {} distinct chains fused-map-verified; associativity held on {}; eq18 precondition held on {}",
        ok_both, assoc_holds, precond_holds
    );
}
