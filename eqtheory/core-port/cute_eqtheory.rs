use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use cute_layout::algebra::mapping::{coalesce, composition};
use cute_layout::layout::Layout;
use cute_layout::tuple::IntTuple::{self, Leaf, Tuple};
use egglog::sort::S;

fn ser_tuple(t: &IntTuple) -> String {
    match t {
        Leaf(n) => n.to_string(),
        Tuple(v) => format!("({})", v.iter().map(ser_tuple).collect::<Vec<_>>().join(" ")),
    }
}

fn ser_layout(l: &Layout) -> String {
    format!("{}|{}", ser_tuple(&l.shape), ser_tuple(&l.stride))
}

fn parse_tuple(s: &str) -> (IntTuple, &str) {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix('(') {
        let mut rest = rest;
        let mut items = Vec::new();
        loop {
            rest = rest.trim_start();
            if let Some(after) = rest.strip_prefix(')') {
                return (Tuple(items), after);
            }
            let (t, r) = parse_tuple(rest);
            items.push(t);
            rest = r;
        }
    } else {
        let end = s
            .find(|c: char| c == '(' || c == ')' || c == ' ')
            .unwrap_or(s.len());
        (Leaf(s[..end].parse().unwrap()), &s[end..])
    }
}

fn parse_layout(s: &str) -> Layout {
    let (shape_s, stride_s) = s.split_once('|').unwrap();
    Layout::new(parse_tuple(shape_s).0, parse_tuple(stride_s).0)
}

fn compose_layouts(a: &str, b: &str) -> Option<String> {
    let c = composition(&parse_layout(a), &parse_layout(b)).ok()?;
    Some(ser_layout(&coalesce(&c)))
}

pub fn register(egraph: &mut egglog::EGraph) {
    egglog::add_primitive!(egraph, "cute-compose" = |a: S, b: S| -?> S {
        compose_layouts(a.as_str(), b.as_str()).map(|s| s.into())
    });
}

const CONSTRUCTORS: &str =
    "(datatype CuteLayout (CuteLit String) (CuteCompose CuteLayout CuteLayout))";

const RULES: &str = concat!(
    "(ruleset cute)\n",
    "(rewrite (CuteCompose (CuteLit ?a) (CuteLit ?b)) (CuteLit (cute-compose ?a ?b)) :ruleset cute)\n"
);

pub fn snippets() -> Vec<EgglogSnippet> {
    vec![
        EgglogSnippet {
            category: SpliceCategory::LogicalConstructors,
            text: CONSTRUCTORS,
        },
        EgglogSnippet {
            category: SpliceCategory::Forward,
            text: RULES,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round(a: i64, b: i64) -> String {
        ser_layout(&Layout::new(
            Tuple(vec![Leaf(a), Leaf(b)]),
            Tuple(vec![Leaf(b), Leaf(1)]),
        ))
    }

    #[test]
    fn cute_compose_closes_a_chain_in_core() {
        let mut eg = crate::egglog_snippet::new_egraph();
        eg.parse_and_run_program(None, &crate::egglog_snippet::assembled_program_for(&[]))
            .unwrap();
        let l1 = round(2, 2048);
        let l2 = round(4, 1024);
        let expected = compose_layouts(&l1, &l2).unwrap();
        let program = format!(
            "(let t (CuteCompose (CuteLit \"{l1}\") (CuteLit \"{l2}\")))\n(run-schedule (saturate cute))\n(check (= t (CuteLit \"{expected}\")))\n"
        );
        eg.parse_and_run_program(None, &program).unwrap();
    }

    fn affine_rows_for_vit_chain() -> usize {
        let mut eg = crate::egglog_snippet::new_egraph();
        eg.parse_and_run_program(None, &crate::egglog_snippet::assembled_program_for(&[]))
            .unwrap();
        eg.parse_and_run_program(None, include_str!("../../repro/subst_layout_blowup.egg"))
            .unwrap();
        eg.parse_and_run_program(None, "(run-schedule (saturate (run prop)))")
            .unwrap();
        for _ in 0..24 {
            eg.parse_and_run_program(
                None,
                "(run-schedule (run main_ruleset) (run prop) (run subst-walk))",
            )
            .unwrap();
        }
        eg.num_tuples()
    }

    fn cute_rows_for_chain(factors: &[i64], n: i64) -> usize {
        let mut term = format!("(CuteLit \"{}\")", round(factors[0], n / factors[0]));
        for &a in &factors[1..] {
            term = format!("(CuteCompose {term} (CuteLit \"{}\"))", round(a, n / a));
        }
        let mut eg = crate::egglog_snippet::new_egraph();
        eg.parse_and_run_program(None, &crate::egglog_snippet::assembled_program_for(&[]))
            .unwrap();
        eg.parse_and_run_program(
            None,
            &format!("(let t {term})\n(run-schedule (saturate cute))\n"),
        )
        .unwrap();
        eg.num_tuples()
    }

    #[test]
    fn head_to_head_vit_chain_rows() {
        let affine = affine_rows_for_vit_chain();
        let factors = [2i64, 4, 8, 16, 32, 64, 128, 2, 4, 8, 16, 32, 64, 128, 2];
        let cute = cute_rows_for_chain(&factors, 4096);
        eprintln!(
            "ViT view chain, e-graph rows: affine theory {affine} vs cute theory {cute} ({}x)",
            affine / cute.max(1)
        );
        assert!(cute * 50 < affine);
    }
}
