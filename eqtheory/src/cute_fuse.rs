use crate::ir::{GraphIr, Nl, NodeKind};

pub const VAL_DATATYPE: &str = r#"
(datatype Val
  (In String)
  (Lit i64)
  (Contract Val Val i64)
  (Mul Val Val)
  (Add Val Val)
  (Sum Val i64)
  (Pt String Val)
  (Gather Val)
  (Scatter Val Val)
  (View Val))
"#;

fn nl_name(nl: Nl) -> &'static str {
    match nl {
        Nl::Exp2 => "exp2",
        Nl::Recip => "recip",
        Nl::Sqrt => "sqrt",
        Nl::Rsqrt => "rsqrt",
        Nl::Relu => "relu",
        Nl::Step => "step",
        Nl::CastF32 => "cast",
    }
}

pub fn term_of(g: &GraphIr, id: usize) -> String {
    let n = |i: usize| format!("n{i}");
    match &g.nodes[id].kind {
        NodeKind::Input { name } => format!("(In \"{name}\")"),
        NodeKind::LiteralF32(_) | NodeKind::LiteralI32(_) => format!("(Lit {id})"),
        NodeKind::Pointwise { nl, read } => format!("(Pt \"{}\" {})", nl_name(*nl), n(read.src)),
        NodeKind::Move { a, b, kext, pair, a_idx, write_idx, write_base } => {
            if let Some(wb) = write_base {
                format!("(Scatter {} {})", n(*wb), n(a.src))
            } else if !a_idx.is_empty() {
                format!("(Gather {})", n(a.src))
            } else if let Some(b) = b {
                if *pair && *kext > 1 {
                    format!("(Contract {} {} {})", n(a.src), n(b.src), kext)
                } else if *pair {
                    format!("(Mul {} {})", n(a.src), n(b.src))
                } else {
                    format!("(Add {} {})", n(a.src), n(b.src))
                }
            } else if *kext > 1 {
                format!("(Sum {} {})", n(a.src), kext)
            } else {
                format!("(View {})", n(a.src))
            }
        }
    }
}

pub fn graph_to_egglog(g: &GraphIr) -> (String, Vec<String>) {
    let mut prog = String::from(VAL_DATATYPE);
    let mut terms = Vec::with_capacity(g.nodes.len());
    for id in 0..g.nodes.len() {
        let t = term_of(g, id);
        prog.push_str(&format!("(let n{id} {t})\n"));
        terms.push(t);
    }
    (prog, terms)
}

fn is_constant(g: &GraphIr, id: usize) -> bool {
    matches!(g.nodes[id].kind, NodeKind::Input { .. } | NodeKind::LiteralF32(_) | NodeKind::LiteralI32(_))
}

fn contraction_key(g: &GraphIr, id: usize) -> Option<(usize, i64)> {
    if let NodeKind::Move { a, b: Some(b), kext, pair: true, a_idx, write_idx, write_base: None } = &g.nodes[id].kind {
        if *kext > 1 && a_idx.is_empty() && write_idx.is_empty() && is_constant(g, b.src) {
            return Some((a.src, *kext));
        }
    }
    None
}

pub fn find_cogrouped_contractions(g: &GraphIr) -> Vec<Vec<usize>> {
    let mut groups: Vec<((usize, i64), Vec<usize>)> = Vec::new();
    for id in 0..g.nodes.len() {
        if let Some(key) = contraction_key(g, id) {
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, ids)) => ids.push(id),
                None => groups.push((key, vec![id])),
            }
        }
    }
    groups.into_iter().filter(|(_, ids)| ids.len() >= 2).map(|(_, ids)| ids).collect()
}

pub fn max_ref(term: &str) -> Option<usize> {
    let mut best: Option<usize> = None;
    let bytes = term.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'n' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let v: usize = term[i + 1..j].parse().unwrap();
            best = Some(best.map_or(v, |b| b.max(v)));
            i = j;
        } else {
            i += 1;
        }
    }
    best
}
