


pub enum HTuple {
    Leaf(i64),
    Tuple(Vec<HTuple>),
}



impl HTuple {

    pub fn size(&self) -> i64 {
        match self {
            HTuple::Leaf(v) => *v,
            HTuple::Tuple(ts) => ts.iter().map(HTuple::size).product(),
        }
    }


    pub fn is_congruent(&self, other: &HTuple) -> bool {
        match (self, other) {
            (HTuple::Leaf(_), HTuple::Leaf(_)) => true,
            (HTuple::Tuple(a), HTuple::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.is_congruent(y))
            },

            _ => false
        }
    }


    pub fn flatten(&self) -> Vec<i64> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }


    pub fn collect_leaves(&self, targ: &mut Vec<i64>) {
        match self {
            HTuple::Leaf(v) => targ.push(*v),
            HTuple::Tuple(ts) => ts.iter().for_each(|t| t.collect_leaves(targ)),
        }
    }


    pub fn rank(&self) -> usize {
        match self {
            HTuple::Leaf(_) => 1,
            HTuple::Tuple(ts) => ts.len(),
        }
    }
}

