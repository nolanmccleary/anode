#[derive(Clone)]
pub enum Op {
    Split(usize, i64),
    Permute(Vec<usize>),
    Merge(usize, usize),
}

#[derive(Clone)]
pub struct Model {
    pub name: String,
    pub dims: Vec<i64>,
    pub ops: Vec<Op>,
}

#[derive(Clone)]
pub struct Modes {
    pub modes: Vec<(i64, i64)>,
}

impl Modes {
    pub fn row_major(dims: &[i64]) -> Self {
        let mut stride = 1i64;
        let mut modes = vec![(0i64, 0i64); dims.len()];
        for i in (0..dims.len()).rev() {
            modes[i] = (dims[i], stride);
            stride *= dims[i];
        }
        Modes { modes }
    }
    pub fn apply(mut self, op: &Op) -> Self {
        match op {
            Op::Split(axis, inner) => {
                let (e, s) = self.modes[*axis];
                self.modes.splice(*axis..=*axis, [(e / inner, inner * s), (*inner, s)]);
                self
            }
            Op::Permute(p) => Modes { modes: p.iter().map(|&i| self.modes[i]).collect() },
            Op::Merge(a, b) => {
                let (e0, s0) = self.modes[*a];
                let (e1, s1) = self.modes[*b];
                assert_eq!(s0, e1 * s1, "cute merge requires contiguous modes");
                self.modes.splice(*a..=*b, [(e0 * e1, s1)]);
                self
            }
        }
    }
    pub fn coalesce(mut self) -> Self {
        let mut i = 0;
        while i + 1 < self.modes.len() {
            let (e0, s0) = self.modes[i];
            let (e1, s1) = self.modes[i + 1];
            if s0 == e1 * s1 {
                self.modes[i] = (e0 * e1, s1);
                self.modes.remove(i + 1);
            } else {
                i += 1;
            }
        }
        self
    }
    pub fn crd2idx(&self, coords: &[i64]) -> i64 {
        self.modes.iter().zip(coords).map(|(&(_, s), &c)| c * s).sum()
    }
    pub fn dims(&self) -> Vec<i64> {
        self.modes.iter().map(|&(e, _)| e).collect()
    }
    pub fn nodes(&self) -> usize {
        self.modes.len() * 2
    }
    pub fn contraction_unit_stride(&self) -> bool {
        self.modes.last().map(|&(_, s)| s == 1).unwrap_or(false)
    }
}

pub fn cute_lower(m: &Model) -> Modes {
    let mut modes = Modes::row_major(&m.dims);
    for op in &m.ops {
        modes = modes.apply(op);
    }
    modes.coalesce()
}

pub fn forward_shapes(m: &Model) -> Vec<Vec<i64>> {
    let mut shapes = vec![m.dims.clone()];
    for op in &m.ops {
        let d = shapes.last().unwrap().clone();
        let nd = match op {
            Op::Split(axis, inner) => {
                let mut nd = d.clone();
                let e = nd[*axis];
                nd.splice(*axis..=*axis, [e / inner, *inner]);
                nd
            }
            Op::Permute(p) => p.iter().map(|&i| d[i]).collect(),
            Op::Merge(a, b) => {
                let mut nd = d.clone();
                let mm = nd[*a] * nd[*b];
                nd.splice(*a..=*b, [mm]);
                nd
            }
        };
        shapes.push(nd);
    }
    shapes
}

pub fn reference_address(m: &Model, final_coords: &[i64]) -> i64 {
    let shapes = forward_shapes(m);
    let mut cur = final_coords.to_vec();
    for (i, op) in m.ops.iter().enumerate().rev() {
        let in_dims = &shapes[i];
        match op {
            Op::Split(axis, inner) => {
                let outer = cur[*axis];
                let inr = cur[*axis + 1];
                cur.splice(*axis..=*axis + 1, [outer * inner + inr]);
            }
            Op::Permute(p) => {
                let mut prev = vec![0i64; p.len()];
                for (new_i, &old_i) in p.iter().enumerate() {
                    prev[old_i] = cur[new_i];
                }
                cur = prev;
            }
            Op::Merge(a, b) => {
                let mm = cur[*a];
                let d_b = in_dims[*b];
                cur.splice(*a..=*a, [mm / d_b, mm % d_b]);
            }
        }
    }
    Modes::row_major(&m.dims).crd2idx(&cur)
}

pub fn decompose(x: i64, shape: &[i64]) -> Vec<i64> {
    let mut coords = vec![0i64; shape.len()];
    let mut rem = x;
    for i in (0..shape.len()).rev() {
        coords[i] = rem % shape[i];
        rem /= shape[i];
    }
    coords
}

pub fn cute_verified(m: &Model) -> bool {
    let cute = cute_lower(m);
    let final_shape = forward_shapes(m).last().unwrap().clone();
    let total: i64 = m.dims.iter().product();
    (0..total.min(4096)).all(|x| {
        cute.crd2idx(&decompose(x, &cute.dims()))
            == reference_address(m, &decompose(x, &final_shape))
    })
}
