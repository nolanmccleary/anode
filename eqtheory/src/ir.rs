use cute_layout::{IntTuple, Layout};

pub type NodeId = usize;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Nl {
    Exp2,
    Recip,
    Sqrt,
    Rsqrt,
    Relu,
    Step,
    CastF32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dt {
    F32,
    I32,
}

#[derive(Clone, Debug)]
pub struct Read {
    pub src: NodeId,
    pub layout: Layout,
    pub base: i64,
    pub nls: Vec<Nl>,
}

impl Read {
    pub fn new(src: NodeId, layout: Layout) -> Self {
        Read { src, layout, base: 0, nls: Vec::new() }
    }
    pub fn at(src: NodeId, layout: Layout, base: i64) -> Self {
        Read { src, layout, base, nls: Vec::new() }
    }
    pub fn with_nls(src: NodeId, layout: Layout, base: i64, nls: Vec<Nl>) -> Self {
        Read { src, layout, base, nls }
    }
}

#[derive(Clone, Debug)]
pub struct IdxRead {
    pub idx: NodeId,
    pub idx_layout: Layout,
    pub stride: i64,
}

#[derive(Clone, Debug)]
pub enum NodeKind {
    Input { name: String },
    LiteralF32(Vec<f32>),
    LiteralI32(Vec<i32>),
    Move {
        a: Read,
        b: Option<Read>,
        kext: i64,
        pair: bool,
        a_idx: Vec<IdxRead>,
        write_idx: Vec<IdxRead>,
        write_base: Option<NodeId>,
        reduce_max: bool,
    },
    Pointwise { nl: Nl, read: Read },
}

#[derive(Clone, Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub dims: Vec<i64>,
    pub dtype: Dt,
}

#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    pub id: NodeId,
}

pub struct GraphIr {
    pub nodes: Vec<Node>,
}

pub fn rm_strides(dims: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        s[k] = s[k + 1] * dims[k + 1];
    }
    s
}

pub fn layout_of(dims: &[i64], strides: &[i64]) -> Layout {
    Layout::new(
        IntTuple::Tuple(dims.iter().map(|d| IntTuple::Leaf(*d)).collect()),
        IntTuple::Tuple(strides.iter().map(|d| IntTuple::Leaf(*d)).collect()),
    )
}

pub fn identity(dims: &[i64]) -> Layout {
    layout_of(dims, &rm_strides(dims))
}

impl GraphIr {
    pub fn new() -> Self {
        GraphIr { nodes: Vec::new() }
    }

    pub fn dims(&self, t: Tensor) -> Vec<i64> {
        self.nodes[t.id].dims.clone()
    }

    fn push(&mut self, kind: NodeKind, dims: Vec<i64>, dtype: Dt) -> Tensor {
        self.nodes.push(Node { kind, dims, dtype });
        Tensor { id: self.nodes.len() - 1 }
    }

    pub fn push_node(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn input(&mut self, name: &str, dims: Vec<i64>, dtype: Dt) -> Tensor {
        self.push(NodeKind::Input { name: name.into() }, dims, dtype)
    }

    pub fn lit_f32(&mut self, vals: Vec<f32>, dims: Vec<i64>) -> Tensor {
        self.push(NodeKind::LiteralF32(vals), dims, Dt::F32)
    }

    pub fn lit_i32(&mut self, vals: Vec<i32>, dims: Vec<i64>) -> Tensor {
        self.push(NodeKind::LiteralI32(vals), dims, Dt::I32)
    }

    pub fn pointwise(&mut self, nl: Nl, x: Tensor) -> Tensor {
        let dims = self.dims(x);
        let read = Read { src: x.id, layout: identity(&dims), base: 0, nls: Vec::new() };
        self.push(NodeKind::Pointwise { nl, read }, dims, Dt::F32)
    }

    pub fn movn(
        &mut self,
        dims: Vec<i64>,
        a: Read,
        b: Option<Read>,
        kext: i64,
        pair: bool,
    ) -> Tensor {
        self.push(
            NodeKind::Move { a, b, kext, pair, a_idx: Vec::new(), write_idx: Vec::new(), write_base: None, reduce_max: false },
            dims,
            Dt::F32,
        )
    }

    pub fn max_last(&mut self, a: Tensor) -> Tensor {
        let dims = self.dims(a);
        let (out, k) = dims.split_at(dims.len() - 1);
        let kext = k[0];
        let st = rm_strides(&dims);
        let ra = Read::new(a.id, layout_of(&dims, &st));
        self.push(
            NodeKind::Move { a: ra, b: None, kext, pair: true, a_idx: Vec::new(), write_idx: Vec::new(), write_base: None, reduce_max: true },
            out.to_vec(),
            Dt::F32,
        )
    }

    pub fn add_bcast_row2(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let dims = self.dims(a);
        let bdims = self.dims(b);
        assert_eq!(&dims[..dims.len() - 1], &bdims[..], "add_bcast2 dims");
        let mut sa = rm_strides(&dims);
        sa.push(0);
        let mut sb = rm_strides(&bdims);
        sb.push(0);
        sb.push(0);
        let mut sh = dims.clone();
        sh.push(2);
        let ra = Read::new(a.id, layout_of(&sh, &sa));
        let rb = Read::new(b.id, layout_of(&sh, &sb));
        self.movn(dims, ra, Some(rb), 2, false)
    }

    pub fn mul(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let dims = self.dims(a);
        assert_eq!(dims, self.dims(b), "mul dims");
        let mut st = rm_strides(&dims);
        st.push(0);
        let mut sh = dims.clone();
        sh.push(1);
        let ra = Read { src: a.id, layout: layout_of(&sh, &st), base: 0, nls: Vec::new() };
        let rb = Read { src: b.id, layout: layout_of(&sh, &st), base: 0, nls: Vec::new() };
        self.movn(dims, ra, Some(rb), 1, true)
    }

    pub fn mul_bcast_row(&mut self, a: Tensor, row: Tensor) -> Tensor {
        let dims = self.dims(a);
        let rdims = self.dims(row);
        assert_eq!(*dims.last().unwrap(), *rdims.last().unwrap(), "bcast width");
        let mut sa = rm_strides(&dims);
        sa.push(0);
        let mut sb = vec![0i64; dims.len() - 1];
        sb.push(1);
        sb.push(0);
        let mut sh = dims.clone();
        sh.push(1);
        let ra = Read { src: a.id, layout: layout_of(&sh, &sa), base: 0, nls: Vec::new() };
        let rb = Read { src: row.id, layout: layout_of(&sh, &sb), base: 0, nls: Vec::new() };
        self.movn(dims, ra, Some(rb), 1, true)
    }

    pub fn mul_bcast_row2(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let dims = self.dims(a);
        let bdims = self.dims(b);
        assert_eq!(&dims[..dims.len() - 1], &bdims[..], "bcast2 dims");
        let mut sa = rm_strides(&dims);
        sa.push(0);
        let mut sb = rm_strides(&bdims);
        sb.push(0);
        sb.push(0);
        let mut sh = dims.clone();
        sh.push(1);
        let ra = Read { src: a.id, layout: layout_of(&sh, &sa), base: 0, nls: Vec::new() };
        let rb = Read { src: b.id, layout: layout_of(&sh, &sb), base: 0, nls: Vec::new() };
        self.movn(dims, ra, Some(rb), 1, true)
    }

    pub fn scale(&mut self, a: Tensor, c: f32) -> Tensor {
        let k = self.lit_f32(vec![c], vec![1]);
        let dims = self.dims(a);
        let mut sa = rm_strides(&dims);
        sa.push(0);
        let mut sb = vec![0i64; dims.len() + 1];
        sb[dims.len()] = 0;
        let mut sh = dims.clone();
        sh.push(1);
        let ra = Read { src: a.id, layout: layout_of(&sh, &sa), base: 0, nls: Vec::new() };
        let rb = Read { src: k.id, layout: layout_of(&sh, &sb), base: 0, nls: Vec::new() };
        self.movn(dims, ra, Some(rb), 1, true)
    }

    pub fn add(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let dims = self.dims(a);
        assert_eq!(dims, self.dims(b), "add dims");
        let mut sh = dims.clone();
        sh.push(2);
        let mut st = rm_strides(&dims);
        st.push(0);
        let ra = Read { src: a.id, layout: layout_of(&sh, &st), base: 0, nls: Vec::new() };
        let rb = Read { src: b.id, layout: layout_of(&sh, &st), base: 0, nls: Vec::new() };
        self.movn(dims, ra, Some(rb), 2, false)
    }

    pub fn add_bcast(&mut self, a: Tensor, s: Tensor) -> Tensor {
        let dims = self.dims(a);
        let mut sh = dims.clone();
        sh.push(2);
        let mut sa = rm_strides(&dims);
        sa.push(0);
        let mut sb = vec![0i64; dims.len() + 1];
        sb[dims.len()] = 0;
        let ra = Read { src: a.id, layout: layout_of(&sh, &sa), base: 0, nls: Vec::new() };
        let rb = Read { src: s.id, layout: layout_of(&sh, &sb), base: 0, nls: Vec::new() };
        self.movn(dims, ra, Some(rb), 2, false)
    }

    pub fn sum_last(&mut self, a: Tensor) -> Tensor {
        let dims = self.dims(a);
        let (out, k) = dims.split_at(dims.len() - 1);
        let kext = k[0];
        let st = rm_strides(&dims);
        let ra = Read { src: a.id, layout: layout_of(&dims, &st), base: 0, nls: Vec::new() };
        self.movn(out.to_vec(), ra, None, kext, true)
    }

    pub fn matmul(&mut self, x: Tensor, w: Tensor) -> Tensor {
        let xd = self.dims(x);
        let wd = self.dims(w);
        let (m, k) = (xd[0], xd[1]);
        assert_eq!(wd[0], k, "matmul inner");
        let n = wd[1];
        let sh = vec![m, n, k];
        let ra = Read { src: x.id, layout: layout_of(&sh, &[k, 0, 1]), base: 0, nls: Vec::new() };
        let rb = Read { src: w.id, layout: layout_of(&sh, &[0, 1, n]), base: 0, nls: Vec::new() };
        self.movn(vec![m, n], ra, Some(rb), k, true)
    }

    pub fn gather_rows(&mut self, table: Tensor, idx: Tensor, out_rows: i64) -> Tensor {
        let td = self.dims(table);
        let width = td[1];
        let dims = vec![out_rows, width];
        let mut sh = dims.clone();
        sh.push(1);
        let ra = Read { src: table.id, layout: layout_of(&sh, &[0, 1, 0]), base: 0, nls: Vec::new() };
        let ir = IdxRead {
            idx: idx.id,
            idx_layout: layout_of(&sh, &[1, 0, 0]),
            stride: width,
        };
        self.push(
            NodeKind::Move {
                a: ra,
                b: None,
                kext: 1,
                pair: true,
                a_idx: vec![ir],
                write_idx: Vec::new(),
                write_base: None,
                reduce_max: false,
            },
            dims,
            Dt::F32,
        )
    }

    pub fn scatter_rows(&mut self, base: Tensor, idx: Tensor, rows: Tensor) -> Tensor {
        let bd = self.dims(base);
        let rd = self.dims(rows);
        let width = bd[1];
        assert_eq!(rd[1], width, "scatter width");
        let mut sh = rd.clone();
        sh.push(1);
        let ra = Read { src: rows.id, layout: layout_of(&sh, &[width, 1, 0]), base: 0, nls: Vec::new() };
        self.scatter_impl(base, idx, ra, width)
    }

    pub fn scatter_rows_sliced(
        &mut self,
        base: Tensor,
        idx: Tensor,
        src: Tensor,
        col_base: i64,
        row_stride: i64,
        rows: i64,
    ) -> Tensor {
        let width = self.dims(base)[1];
        let sh = vec![rows, width, 1];
        let ra = Read::at(src.id, layout_of(&sh, &[row_stride, 1, 0]), col_base);
        self.scatter_impl(base, idx, ra, width)
    }

    fn scatter_impl(&mut self, base: Tensor, idx: Tensor, ra: Read, width: i64) -> Tensor {
        let bd = self.dims(base);
        let sh = vec![self.nodes[ra.src].dims.first().copied().unwrap_or(1), width, 1];
        let wi = IdxRead {
            idx: idx.id,
            idx_layout: layout_of(&sh, &[1, 0, 0]),
            stride: width,
        };
        self.push(
            NodeKind::Move {
                a: ra,
                b: None,
                kext: 1,
                pair: true,
                a_idx: Vec::new(),
                write_idx: vec![wi],
                write_base: Some(base.id),
                reduce_max: false,
            },
            bd,
            Dt::F32,
        )
    }
}
