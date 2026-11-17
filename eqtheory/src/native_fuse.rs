use crate::ir::{GraphIr, Nl, NodeKind};
use egglog::{CommandOutput, EGraph, Term};
use std::collections::{HashMap, HashSet};

pub struct Fusions {
    pub claims: HashSet<usize>,
    pub rms: HashMap<usize, (usize, usize)>,
    pub silu: HashMap<usize, (usize, usize, usize, usize)>,
    pub attn: HashMap<usize, (usize, usize, usize, usize, usize, usize)>,
}

fn consumers(g: &GraphIr) -> Vec<i64> {
    let mut c = vec![0i64; g.nodes.len()];
    for node in &g.nodes {
        match &node.kind {
            NodeKind::Move { a, b, a_idx, write_idx, write_base, .. } => {
                c[a.src] += 1;
                if let Some(b) = b {
                    c[b.src] += 1;
                }
                for i in a_idx {
                    c[i.idx] += 1;
                }
                for i in write_idx {
                    c[i.idx] += 1;
                }
                if let Some(wb) = write_base {
                    c[*wb] += 1;
                }
            }
            NodeKind::Pointwise { read, .. } => c[read.src] += 1,
            _ => {}
        }
    }
    c
}

pub fn plan(g: &GraphIr) -> Fusions {
    let cons = consumers(g);
    let mut facts = String::new();
    for (id, node) in g.nodes.iter().enumerate() {
        let i = id as i64;
        facts.push_str(&format!("(consumers {i} {})\n", cons[id]));
        match &node.kind {
            NodeKind::LiteralF32(v) if v.len() == 1 => facts.push_str(&format!("(lit {i})\n")),
            NodeKind::Pointwise { nl, read } => {
                let name = match nl {
                    Nl::Exp2 => "exp2",
                    Nl::Recip => "recip",
                    Nl::Rsqrt => "rsqrt",
                    Nl::Sqrt => "sqrt",
                    Nl::Relu => "relu",
                    Nl::Step => "step",
                    Nl::CastF32 => "cast",
                };
                facts.push_str(&format!("(pw {i} \"{name}\" {})\n", read.src));
            }
            NodeKind::Move { a, b, kext, pair, a_idx, write_idx, .. } => {
                if a_idx.is_empty() && write_idx.is_empty() {
                    match b {
                        Some(b) => facts.push_str(&format!(
                            "(mv2 {i} {} {} {} {})\n",
                            a.src,
                            b.src,
                            kext,
                            if *pair { 1 } else { 0 }
                        )),
                        None => facts.push_str(&format!("(mv1 {i} {} {})\n", a.src, kext)),
                    }
                }
            }
            _ => {}
        }
    }
    let program = format!(
        "{}{}{}",
        "(relation consumers (i64 i64))\n(relation lit (i64))\n(relation pw (i64 String i64))\n(relation mv2 (i64 i64 i64 i64 i64))\n(relation mv1 (i64 i64 i64))\n(relation rms (i64 i64 i64))\n(relation silu (i64 i64 i64 i64 i64))\n(relation attn (i64 i64 i64 i64 i64 i64 i64))\n(relation claim (i64))\n",
        facts,
        "(rule ((mv2 ?x2 ?x ?x 1 1) (consumers ?x2 1) (mv1 ?s ?x2 ?d) (> ?d 1) (consumers ?s 1) (mv2 ?ms ?s ?c1 1 1) (lit ?c1) (consumers ?ms 1) (mv2 ?mse ?ms ?ce 2 0) (lit ?ce) (consumers ?mse 1) (pw ?r \"rsqrt\" ?mse) (consumers ?r 1) (mv2 ?out ?x ?r 1 1)) ((rms ?out ?x ?ce) (claim ?x2) (claim ?s) (claim ?ms) (claim ?mse) (claim ?r)))\n(rule ((mv2 ?neg ?g ?cl 1 1) (lit ?cl) (consumers ?neg 1) (pw ?e \"exp2\" ?neg) (consumers ?e 1) (mv2 ?den ?e ?c1 2 0) (lit ?c1) (consumers ?den 1) (pw ?sig \"recip\" ?den) (consumers ?sig 1) (mv2 ?act ?g ?sig 1 1) (consumers ?act 1) (mv2 ?h ?act ?up 1 1)) ((silu ?h ?g ?up ?cl ?c1) (claim ?neg) (claim ?e) (claim ?den) (claim ?sig) (claim ?act)))\n(rule ((mv2 ?sc ?q ?kc ?hd 1) (> ?hd 1) (consumers ?sc 1) (mv2 ?scaled ?sc ?cs 1 1) (lit ?cs) (consumers ?scaled 1) (mv2 ?biased ?scaled ?mask 2 0) (consumers ?biased 1) (mv2 ?l2 ?biased ?cl 1 1) (lit ?cl) (consumers ?l2 1) (pw ?ex \"exp2\" ?l2) (consumers ?ex 2) (mv1 ?den ?ex ?ctx) (consumers ?den 1) (pw ?rd \"recip\" ?den) (consumers ?rd 1) (mv2 ?probs ?ex ?rd 1 1) (consumers ?probs 1) (mv2 ?av ?probs ?v ?ctx 1)) ((attn ?av ?q ?kc ?v ?mask ?cs ?sc) (claim ?sc) (claim ?scaled) (claim ?biased) (claim ?l2) (claim ?ex) (claim ?den) (claim ?rd) (claim ?probs)))\n(run 2)\n"
    );
    let mut eg = EGraph::default();
    eg.parse_and_run_program(None, &program).expect("native fusion egglog");
    let rows = |eg: &mut EGraph, name: &str, arity: usize| -> Vec<Vec<i64>> {
        let out = eg
            .parse_and_run_program(None, &format!("(print-function {name} 1000000)"))
            .expect("print");
        let mut res = Vec::new();
        for o in out {
            if let CommandOutput::PrintFunction(_f, dag, rows, _m) = o {
                for (key, _) in rows {
                    if let Term::App(nm, children) = dag.get(key) {
                        if nm.as_str() == name && children.len() == arity {
                            let kids = children.clone();
                            res.push(
                                (0..arity)
                                    .map(|k| dag.to_string(kids[k]).parse::<i64>().unwrap())
                                    .collect(),
                            );
                        }
                    }
                }
            }
        }
        res
    };
    let mut f = Fusions {
        claims: HashSet::new(),
        rms: HashMap::new(),
        silu: HashMap::new(),
        attn: HashMap::new(),
    };
    for r in rows(&mut eg, "claim", 1) {
        f.claims.insert(r[0] as usize);
    }
    for r in rows(&mut eg, "rms", 3) {
        f.rms.insert(r[0] as usize, (r[1] as usize, r[2] as usize));
    }
    for r in rows(&mut eg, "silu", 5) {
        f.silu.insert(
            r[0] as usize,
            (r[1] as usize, r[2] as usize, r[3] as usize, r[4] as usize),
        );
    }
    for r in rows(&mut eg, "attn", 7) {
        f.attn.insert(
            r[0] as usize,
            (
                r[1] as usize,
                r[2] as usize,
                r[3] as usize,
                r[4] as usize,
                r[5] as usize,
                r[6] as usize,
            ),
        );
    }
    f
}
