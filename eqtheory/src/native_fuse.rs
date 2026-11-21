use crate::ir::GraphIr;
use std::collections::{HashMap, HashSet};

pub struct Fusions {
    pub claims: HashSet<usize>,
    pub rms: HashMap<usize, (usize, usize)>,
    pub silu: HashMap<usize, (usize, usize, usize, usize)>,
    pub attn: HashMap<usize, (usize, usize, usize, usize, usize, usize)>,
}

pub fn plan(_g: &GraphIr) -> Fusions {
    Fusions {
        claims: HashSet::new(),
        rms: HashMap::new(),
        silu: HashMap::new(),
        attn: HashMap::new(),
    }
}
