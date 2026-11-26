use crate::ir::{GraphIr, Nl, NodeKind};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Semimodule {
    Additive,
    Multiplicative,
    Tropical,
}

pub fn needs_rational_scalar(nl: Nl) -> bool {
    matches!(nl, Nl::Rsqrt | Nl::Sqrt)
}

pub fn tag_nodes(g: &GraphIr) -> Vec<Semimodule> {
    let mut tag = vec![Semimodule::Additive; g.nodes.len()];
    for id in 0..g.nodes.len() {
        tag[id] = match &g.nodes[id].kind {
            NodeKind::Input { .. } | NodeKind::LiteralF32(_) | NodeKind::LiteralI32(_) => {
                Semimodule::Additive
            }
            NodeKind::Pointwise { nl, read } => match nl {
                Nl::Exp2 => Semimodule::Multiplicative,
                Nl::Relu | Nl::Step => Semimodule::Tropical,
                _ => tag[read.src],
            },
            NodeKind::Move { a, b, kext, pair, a_idx, write_idx, write_base, reduce_max: _ } => {
                let contraction = b.is_some()
                    && *pair
                    && *kext > 1
                    && a_idx.is_empty()
                    && write_idx.is_empty()
                    && write_base.is_none();
                if contraction {
                    Semimodule::Additive
                } else {
                    tag[a.src]
                }
            }
        };
    }
    tag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_model::{build, NativeCfg};

    #[test]
    fn softmax_region_is_multiplicative_rest_additive() {
        let cfg = NativeCfg {
            vocab: 100, d: 16, ff: 32, n_heads: 2, n_kv_heads: 1,
            layers: 1, cap: 4, ctx: 4, seq: 4, prev: 0,
        };
        let nm = build(&cfg, (0..4).collect());
        let tag = tag_nodes(&nm.g);

        let n_mult = tag.iter().filter(|t| **t == Semimodule::Multiplicative).count();
        assert!(n_mult > 0, "exp2-conjugated (softmax/sigmoid) regions must tag Multiplicative");

        for (name, id) in &nm.probes {
            if name.starts_with("resid") || name.starts_with("keys") || name.starts_with("attn_h") {
                assert_eq!(tag[*id], Semimodule::Additive, "{name} is post-contraction and must be Additive");
            }
        }

        for (id, node) in nm.g.nodes.iter().enumerate() {
            if let NodeKind::Pointwise { nl: Nl::Exp2, .. } = &node.kind {
                assert_eq!(tag[id], Semimodule::Multiplicative, "exp2 node {id} enters Multiplicative");
            }
        }
    }
}
