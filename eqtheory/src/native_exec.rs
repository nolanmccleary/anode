use crate::ir::{Dt, GraphIr, IdxRead, Nl, NodeKind, Read};
use crate::metal_backend::CuteMetal;
use cute_layout::{IntTuple, Layout};
use metal::Buffer;

fn leaves(t: &IntTuple) -> Vec<i64> {
    match t {
        IntTuple::Leaf(v) => vec![*v],
        IntTuple::Tuple(v) => v.iter().flat_map(leaves).collect(),
    }
}

fn axes_of(l: &Layout) -> (Vec<i64>, Vec<i64>) {
    (leaves(&l.shape), leaves(&l.stride))
}

fn meta_from_axes(sh: &[i64], st: &[i64], take: usize) -> Option<[i32; 18]> {
    let mut modes: Vec<(i64, i64)> = Vec::new();
    for i in (0..take).rev() {
        if sh[i] != 1 {
            modes.push((sh[i], st[i]));
        }
    }
    if modes.is_empty() {
        modes.push((1, 0));
    }
    if modes.len() > 8 {
        return None;
    }
    let mut m = [0i32; 18];
    m[0] = modes.len() as i32;
    for (k, (s, d)) in modes.iter().enumerate() {
        m[2 + k] = i32::try_from(*s).ok()?;
        m[10 + k] = i32::try_from(*d).ok()?;
    }
    Some(m)
}

fn meta_of(l: &Layout) -> Option<[i32; 18]> {
    let (sh, st) = axes_of(l);
    meta_from_axes(&sh, &st, sh.len())
}

fn flat_meta(n: i64) -> [i32; 18] {
    let mut m = [0i32; 18];
    m[0] = 1;
    m[2] = n as i32;
    m[10] = 1;
    m
}

fn numel(d: &[i64]) -> i64 {
    d.iter().product()
}

fn nl_c(nl: Nl, x: &str) -> String {
    match nl {
        Nl::Exp2 => format!("exp2({x})"),
        Nl::Recip => format!("(1.0f/({x}))"),
        Nl::Sqrt => format!("sqrt({x})"),
        Nl::Rsqrt => format!("rsqrt({x})"),
        Nl::Relu => format!("max({x}, 0.0f)"),
        Nl::Step => format!("(({x}) > 0.0f ? 1.0f : 0.0f)"),
        Nl::CastF32 => format!("({x})"),
    }
}

fn opref(
    read: &Read,
    inset: &std::collections::HashSet<usize>,
    ext: &mut Vec<(usize, [i32; 18])>,
    slot: &mut std::collections::HashMap<String, usize>,
) -> String {
    if inset.contains(&read.src) {
        return format!("v{}", read.src);
    }
    let (sh, st) = axes_of(&read.layout);
    let take = if st.last() == Some(&0) { sh.len().saturating_sub(1) } else { sh.len() };
    let meta = meta_from_axes(&sh, &st, take).unwrap_or_else(|| flat_meta(1));
    let key = format!("{}:{:?}", read.src, &meta[..]);
    let k = if let Some(k) = slot.get(&key) {
        *k
    } else {
        let k = ext.len();
        ext.push((read.src, meta));
        slot.insert(key, k);
        k
    };
    format!("in{k}[strided_off(m{k}, gid)]")
}

#[allow(clippy::too_many_arguments)]
fn gen_operand(
    g: &GraphIr,
    read: &Read,
    internal: &std::collections::HashSet<usize>,
    special: &std::collections::HashMap<usize, String>,
    idx: &str,
    ext: &mut Vec<(usize, [i32; 18])>,
    slot: &mut std::collections::HashMap<String, usize>,
) -> String {
    if let Some(v) = special.get(&read.src) {
        return v.clone();
    }
    if internal.contains(&read.src) {
        return gen_node(g, read.src, internal, special, idx, ext, slot);
    }
    let (sh, st) = axes_of(&read.layout);
    let take = if st.last() == Some(&0) { sh.len().saturating_sub(1) } else { sh.len() };
    let meta = meta_from_axes(&sh, &st, take).unwrap_or_else(|| flat_meta(1));
    let key = format!("{}:{:?}", read.src, &meta[..]);
    let k = if let Some(k) = slot.get(&key) {
        *k
    } else {
        let k = ext.len();
        ext.push((read.src, meta));
        slot.insert(key, k);
        k
    };
    format!("in{k}[strided_off(m{k}, {idx})]")
}

#[allow(clippy::too_many_arguments)]
fn gen_node(
    g: &GraphIr,
    id: usize,
    internal: &std::collections::HashSet<usize>,
    special: &std::collections::HashMap<usize, String>,
    idx: &str,
    ext: &mut Vec<(usize, [i32; 18])>,
    slot: &mut std::collections::HashMap<String, usize>,
) -> String {
    match &g.nodes[id].kind {
        NodeKind::Pointwise { nl, read } => {
            let o = gen_operand(g, read, internal, special, idx, ext, slot);
            nl_c(*nl, &o)
        }
        NodeKind::Move { a, b, pair, .. } => {
            let ae = gen_operand(g, a, internal, special, idx, ext, slot);
            match b {
                Some(b) => {
                    let be = gen_operand(g, b, internal, special, idx, ext, slot);
                    if *pair { format!("({ae} * {be})") } else { format!("({ae} + {be})") }
                }
                None => ae,
            }
        }
        _ => "0.0f".into(),
    }
}

fn gen_region_body(g: &GraphIr, region: &[usize]) -> (String, String, Vec<(usize, [i32; 18])>) {
    let inset: std::collections::HashSet<usize> = region.iter().copied().collect();
    let mut ext: Vec<(usize, [i32; 18])> = Vec::new();
    let mut slot: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut nodes = region.to_vec();
    nodes.sort_unstable();
    let mut body = String::new();
    for &id in &nodes {
        let e = match &g.nodes[id].kind {
            NodeKind::Pointwise { nl, read } => {
                let o = opref(read, &inset, &mut ext, &mut slot);
                nl_c(*nl, &o)
            }
            NodeKind::Move { a, b, pair, .. } => {
                let ae = opref(a, &inset, &mut ext, &mut slot);
                match b {
                    Some(b) => {
                        let be = opref(b, &inset, &mut ext, &mut slot);
                        if *pair { format!("{ae} * {be}") } else { format!("{ae} + {be}") }
                    }
                    None => ae,
                }
            }
            _ => "0.0f".into(),
        };
        body += &format!("    float v{id} = {e};\n");
    }
    let b = 1 + ext.len();
    let mut hdr = String::from("    device float* OUT [[buffer(0)]],\n");
    for k in 0..ext.len() {
        hdr += &format!("    device const float* in{k} [[buffer({})]],\n", 1 + k);
    }
    hdr += &format!("    constant uint& n [[buffer({b})]],\n");
    for k in 0..ext.len() {
        hdr += &format!("    constant int* m{k} [[buffer({})]],\n", b + 1 + k);
    }
    (hdr, body, ext)
}

enum Tape {
    Run1d {
        kernel: &'static str,
        bufs: Vec<Buffer>,
        offsets: Vec<u64>,
        metas: Vec<(u64, [i32; 18])>,
        scalars: Vec<(u64, u32)>,
        specs: Vec<(u64, [i32; 8])>,
        progs: Vec<(u64, [i32; 25])>,
        n: u64,
    },
    Kernel {
        kidx: usize,
        variant: crate::kernelset::Variant,
        bufs: Vec<Buffer>,
        comp: crate::kernelset::Composition,
    },
}

#[derive(Clone)]
pub struct ContractShape {
    pub m: usize,
    pub n: usize,
    pub ap: [i32; 4],
    pub bp: [i32; 4],
    pub cp: [i32; 4],
    pub batch: Vec<i32>,
    pub nbatch: i64,
}

pub fn classify_contract(a: &Read, b: &Read, kext: i64, dims: &[i64]) -> ContractShape {
    let (sha, sta) = axes_of(&a.layout);
    let (_shb, stb) = axes_of(&b.layout);
    let rank = sha.len();
    let kax = rank - 1;
    assert_eq!(sha[kax], kext, "k axis extent: a.sha={sha:?} a.sta={sta:?} b.sta={stb:?} kext={kext} dims={dims:?}");
    let mut a_only: Vec<usize> = Vec::new();
    let mut b_only: Vec<usize> = Vec::new();
    let mut b_axes: Vec<usize> = Vec::new();
    for ax in 0..kax {
        if dims[ax] == 1 {
            continue;
        }
        match (sta[ax] != 0, stb[ax] != 0) {
            (true, false) => a_only.push(ax),
            (false, true) => b_only.push(ax),
            _ => b_axes.push(ax),
        }
    }
    let m_ax = a_only.pop();
    let n_ax = b_only.pop();
    b_axes.extend(a_only);
    b_axes.extend(b_only);
    b_axes.sort_unstable();
    let after = |ax: usize| -> i64 { dims[ax + 1..].iter().product() };
    let (m, am_s, cm_s) = match m_ax {
        Some(ax) => (dims[ax], sta[ax], after(ax)),
        None => (1, 0, 0),
    };
    let (nn, bn_s, cn_s) = match n_ax {
        Some(ax) => (dims[ax], stb[ax], after(ax)),
        None => (1, 0, 0),
    };
    let mut batch = vec![b_axes.len() as i32];
    let mut nbatch = 1i64;
    for ax in &b_axes {
        batch.extend([
            dims[*ax] as i32,
            sta[*ax] as i32,
            stb[*ax] as i32,
            after(*ax) as i32,
        ]);
        nbatch *= dims[*ax];
    }
    if let (Some(ma), Some(na)) = (m_ax, n_ax) {
        if let Some(maxb) = b_axes.iter().max() {
            assert!(*maxb < ma.min(na), "batch axes must be outermost");
        }
    }
    assert_eq!(numel(dims), m * nn * nbatch, "contract extents");
    let crs = if m_ax.is_some() { cm_s } else { 1 };
    let ccs = if n_ax.is_some() { cn_s } else { 1 };
    ContractShape {
        m: m as usize,
        n: nn as usize,
        ap: [0, am_s as i32, sta[kax] as i32, 0],
        bp: [0, stb[kax] as i32, bn_s as i32, 0],
        cp: [0, crs as i32, ccs as i32, 0],
        batch,
        nbatch,
    }
}

fn pure_copy_read(g: &GraphIr, id: usize) -> Option<&Read> {
    if let NodeKind::Move { a, b: None, kext: 1, a_idx, write_idx, .. } = &g.nodes[id].kind {
        if a_idx.is_empty() && write_idx.is_empty() && g.nodes[id].dtype == g.nodes[a.src].dtype {
            return Some(a);
        }
    }
    None
}

fn resolve_read(g: &GraphIr, hazard: &std::collections::HashSet<usize>, r: &Read) -> Read {
    let mut cur = r.clone();
    loop {
        let Some(a) = pure_copy_read(g, cur.src) else { break };
        if hazard.contains(&a.src) {
            break;
        }
        let (ash, ast) = axes_of(&a.layout);
        let rsh: Vec<i64> = ash.iter().rev().cloned().collect();
        let rst: Vec<i64> = ast.iter().rev().cloned().collect();
        let af = crate::ir::layout_of(&rsh, &rst);
        let Ok(comp) = cute_layout::algebra::composition(&af, &cur.layout) else {
            break;
        };
        let (bsh, _) = axes_of(&cur.layout);
        let (csh, _) = axes_of(&comp);
        if bsh != csh {
            break;
        }
        cur = Read { src: a.src, layout: comp, base: 0, nls: cur.nls.clone() };
    }
    cur
}

fn pack_nls(nls: &[Nl]) -> u32 {
    assert!(nls.len() <= 8, "nl-chain longer than 8 not packable");
    let mut ops = 0u32;
    for (k, nl) in nls.iter().enumerate() {
        let op: u32 = match nl {
            Nl::Exp2 => 1,
            Nl::Recip => 2,
            Nl::Sqrt => 3,
            Nl::Rsqrt => 4,
            Nl::Relu => 5,
            Nl::Step => 6,
            Nl::CastF32 => 7,
        };
        ops |= op << (4 * k as u32);
    }
    ops
}

fn eval_flat(sh: &[i64], st: &[i64], i: i64) -> i64 {
    let mut r = i;
    let mut off = 0i64;
    for k in (0..sh.len()).rev() {
        off += (r % sh[k]) * st[k];
        r /= sh[k];
    }
    off
}

fn identity_move(g: &GraphIr, node: &crate::ir::Node, a: &Read, a_idx: &[IdxRead], n: i64) -> bool {
    let src = &g.nodes[a.src];
    if node.dtype != src.dtype || n > numel(&src.dims) {
        return false;
    }
    let mut idx_vals: Vec<(&[i32], Vec<i64>, Vec<i64>, i64)> = Vec::new();
    for ir in a_idx {
        let NodeKind::LiteralI32(vals) = &g.nodes[ir.idx].kind else {
            return false;
        };
        let (ish, ist) = axes_of(&ir.idx_layout);
        idx_vals.push((vals, ish, ist, ir.stride));
    }
    let (sh, st) = axes_of(&a.layout);
    for i in 0..n {
        let mut off = eval_flat(&sh, &st, i);
        for (vals, ish, ist, stride) in &idx_vals {
            off += vals[eval_flat(ish, ist, i) as usize] as i64 * stride;
        }
        if off != i {
            return false;
        }
    }
    true
}

pub struct NativeExec {
    pub backend: CuteMetal,
    bufs: Vec<Option<Buffer>>,
    tape: Vec<Tape>,
    pub dispatches: usize,
    pub alloc_total: usize,
    pub inplace_cache: bool,
    pub elided: usize,
    input_ids: std::collections::HashSet<usize>,
    hazard: std::collections::HashSet<usize>,
    pending: std::collections::HashMap<usize, (Read, i64)>,
    fused_names: std::collections::HashMap<u64, &'static str>,
    pub dead: std::collections::HashSet<usize>,
}

impl NativeExec {
    pub fn new() -> Self {
        NativeExec {
            backend: CuteMetal::new(),
            bufs: Vec::new(),
            tape: Vec::new(),
            dispatches: 0,
            alloc_total: 0,
            inplace_cache: false,
            elided: 0,
            input_ids: Default::default(),
            hazard: Default::default(),
            pending: Default::default(),
            fused_names: Default::default(),
            dead: Default::default(),
        }
    }

    fn use_buf(&mut self, id: usize) -> Buffer {
        if self.bufs[id].is_none() {
            if let Some((r, n)) = self.pending.remove(&id) {
                self.emit_plain_copy(id, &r, n);
            }
        }
        self.buf(id)
    }

    fn guard_write_overlap(&self, node: usize, write: &Buffer, reads: &[&Buffer]) {
        let wp = write.contents() as usize;
        let wl = write.length() as usize;
        for r in reads {
            let rp = r.contents() as usize;
            let rl = r.length() as usize;
            if wp < rp + rl && rp < wp + wl {
                panic!("node {node}: move writes into memory it reads (write {wp:#x}+{wl}, read {rp:#x}+{rl})");
            }
        }
    }

    fn emit_plain_copy(&mut self, id: usize, a: &Read, n: i64) {
        assert!(a.nls.is_empty(), "copy_strided cannot apply inline nls");
        let meta = meta_of(&a.layout).expect("copy meta");
        let out = self.alloc(n as usize * 4);
        let sb = self.use_buf(a.src);
        self.guard_write_overlap(id, &out, &[&sb]);
        self.xcopy(sb, out.clone(), vec![], n, meta, flat_meta(n), [0; 8], [0; 8], 0);
        self.bufs[id] = Some(out);
    }

    pub fn poke_f32(&self, id: usize, vals: &[f32]) {
        assert!(self.input_ids.contains(&id), "poke target {id} is not an Input");
        let b = self.buf(id);
        unsafe {
            std::ptr::copy_nonoverlapping(
                vals.as_ptr() as *const u8,
                b.contents() as *mut u8,
                vals.len() * 4,
            );
        }
    }

    pub fn poke_i32(&self, id: usize, vals: &[i32]) {
        assert!(self.input_ids.contains(&id), "poke target {id} is not an Input");
        let b = self.buf(id);
        unsafe {
            std::ptr::copy_nonoverlapping(
                vals.as_ptr() as *const u8,
                b.contents() as *mut u8,
                vals.len() * 4,
            );
        }
    }

    fn alloc(&mut self, bytes: usize) -> Buffer {
        self.alloc_total += bytes;
        self.backend.alloc_bytes(bytes)
    }

    fn buf(&self, id: usize) -> Buffer {
        self.bufs[id].clone().expect("buffer ready")
    }

    fn push1d(
        &mut self,
        kernel: &'static str,
        bufs: Vec<Buffer>,
        metas: Vec<(u64, [i32; 18])>,
        scalars: Vec<(u64, u32)>,
        specs: Vec<(u64, [i32; 8])>,
        n: u64,
    ) {
        self.tape.push(Tape::Run1d { kernel, bufs, offsets: Vec::new(), metas, scalars, specs, progs: Vec::new(), n });
        self.dispatches += 1;
    }

    fn push1d_off(
        &mut self,
        kernel: &'static str,
        bufs: Vec<Buffer>,
        offsets: Vec<u64>,
        metas: Vec<(u64, [i32; 18])>,
        scalars: Vec<(u64, u32)>,
        specs: Vec<(u64, [i32; 8])>,
        n: u64,
    ) {
        self.tape.push(Tape::Run1d { kernel, bufs, offsets, metas, scalars, specs, progs: Vec::new(), n });
        self.dispatches += 1;
    }

    fn xcopy(
        &mut self,
        src: Buffer,
        dst: Buffer,
        idx: Vec<(Buffer, [i32; 18])>,
        n: i64,
        rmeta: [i32; 18],
        wmeta: [i32; 18],
        rspec: [i32; 8],
        wspec: [i32; 8],
        src_base: i64,
    ) {
        let d = |k: usize, idx: &Vec<(Buffer, [i32; 18])>| {
            idx.get(k).map(|(b, _)| b.clone()).unwrap_or_else(|| src.clone())
        };
        let (i0, i1, i2, i3) = (d(0, &idx), d(1, &idx), d(2, &idx), d(3, &idx));
        let im = |k: usize| idx.get(k).map(|(_, m)| *m).unwrap_or([0i32; 18]);
        let metas = vec![(7, rmeta), (8, wmeta), (11, im(0)), (12, im(1)), (13, im(2)), (14, im(3))];
        let scalars = vec![(6, n as u32)];
        let specs = vec![(9, rspec), (10, wspec)];
        if src_base != 0 {
            self.push1d_off(
                "xcopy_f",
                vec![src, dst, i0, i1, i2, i3],
                vec![(src_base * 4) as u64, 0, 0, 0, 0, 0],
                metas,
                scalars,
                specs,
                n as u64,
            );
        } else {
            self.push1d("xcopy_f", vec![src, dst, i0, i1, i2, i3], metas, scalars, specs, n as u64);
        }
    }

    pub fn build(&mut self, g: &GraphIr, inputs: &std::collections::HashMap<String, Vec<f32>>) {
        let coarsen = std::env::var("NOCOARSEN").is_err();
        let fused = (!coarsen && std::env::var("NOFUSE").is_err()).then(|| crate::compose::refine_pointwise(g));
        let g = fused.as_ref().unwrap_or(g);
        self.build_inner(g, inputs);
    }

    fn build_inner(&mut self, g: &GraphIr, inputs: &std::collections::HashMap<String, Vec<f32>>) {
        self.bufs = vec![None; g.nodes.len()];
        self.tape.clear();
        self.pending.clear();
        self.input_ids.clear();
        self.dispatches = 0;
        self.alloc_total = 0;
        self.elided = 0;
        self.hazard = Default::default();
        if self.inplace_cache {
            for node in g.nodes.iter() {
                if let NodeKind::Move { write_idx, write_base: Some(wb), .. } = &node.kind {
                    if !write_idx.is_empty() && matches!(g.nodes[*wb].kind, NodeKind::Input { .. }) {
                        self.hazard.insert(*wb);
                    }
                }
            }
        }
        let mut referenced = vec![false; g.nodes.len()];
        for node in g.nodes.iter() {
            match &node.kind {
                NodeKind::Move { a, b, a_idx, write_idx, write_base, .. } => {
                    referenced[a.src] = true;
                    if let Some(b) = b {
                        referenced[b.src] = true;
                    }
                    for i in a_idx.iter().chain(write_idx.iter()) {
                        referenced[i.idx] = true;
                    }
                    if let Some(wb) = write_base {
                        referenced[*wb] = true;
                    }
                }
                NodeKind::Pointwise { read, .. } => referenced[read.src] = true,
                _ => {}
            }
        }
        let coarsen = std::env::var("NOCOARSEN").is_err();
        let mut partition: std::collections::HashMap<usize, (crate::extract::Servicer, Vec<usize>)> =
            Default::default();
        let mut kskip: std::collections::HashSet<usize> = Default::default();
        if coarsen {
            let outputs: Vec<usize> = (0..g.nodes.len()).collect();
            for k in crate::extract::partition(g, &outputs) {
                for &n in &k.nodes {
                    if n != k.out {
                        kskip.insert(n);
                    }
                }
                partition.insert(k.out, (k.servicer, k.nodes));
            }
        }
        for (id, node) in g.nodes.iter().enumerate() {
            if self.dead.contains(&id) {
                self.elided += 1;
                continue;
            }
            if coarsen {
                if let Some((servicer, nodes)) = partition.get(&id) {
                    match servicer {
                        crate::extract::Servicer::Flash { scores, mask, .. } => {
                            let hd = match &g.nodes[*scores].kind {
                                NodeKind::Move { kext, .. } => *kext,
                                _ => 1,
                            };
                            let scale = 1.0f32 / (hd as f32).sqrt();
                            let (sc, mask) = (*scores, *mask);
                            self.emit_attn(g, id, 0, 0, 0, mask, scale, sc);
                            continue;
                        }
                        crate::extract::Servicer::Reduce(rr) => {
                            self.emit_reduce_region(g, &rr.clone());
                            continue;
                        }
                        crate::extract::Servicer::Elementwise => {
                            self.emit_fused_region(g, &nodes.clone(), id);
                            continue;
                        }
                        crate::extract::Servicer::Contraction | crate::extract::Servicer::Primitive => {}
                    }
                } else if kskip.contains(&id) {
                    self.elided += 1;
                    continue;
                }
            }
            if matches!(node.kind, NodeKind::Pointwise { .. }) && !referenced[id] {
                self.elided += 1;
                continue;
            }
            let dims = node.dims.clone();
            let n = numel(&dims);
            match &node.kind {
                NodeKind::Input { name } => {
                    let host = inputs
                        .get(name)
                        .unwrap_or_else(|| panic!("missing input {name}"));
                    assert_eq!(host.len() as i64, n, "input {name} size");
                    self.input_ids.insert(id);
                    self.bufs[id] = Some(match node.dtype {
                        Dt::I32 => {
                            let iv: Vec<i32> = host.iter().map(|v| *v as i32).collect();
                            self.backend.upload_i32(&iv)
                        }
                        Dt::F32 => self.backend.upload(host),
                    });
                }
                NodeKind::LiteralF32(v) => {
                    self.bufs[id] = Some(self.backend.upload(v));
                }
                NodeKind::LiteralI32(v) => {
                    self.bufs[id] = Some(self.backend.upload_i32(v));
                }
                NodeKind::Pointwise { nl, read } => {
                    let kernel = match nl {
                        Nl::Exp2 => "exp2_s",
                        Nl::Recip => "recip_s",
                        Nl::Sqrt => "sqrt_s",
                        Nl::Rsqrt => "rsqrt_s",
                        other => panic!("pointwise {other:?} unused by this model"),
                    };
                    let rr = resolve_read(g, &self.hazard, read);
                    assert!(rr.nls.is_empty(), "pointwise _s kernels cannot apply inline nls");
                    let meta = meta_of(&rr.layout).expect("pointwise meta");
                    let out = self.alloc(n as usize * 4);
                    let sb = self.use_buf(rr.src);
                    self.guard_write_overlap(id, &out, &[&sb]);
                    self.push1d_off(
                        kernel,
                        vec![sb, out.clone()],
                        vec![(rr.base * 4) as u64, 0],
                        vec![(3, meta)],
                        vec![(2, n as u32)],
                        vec![],
                        n as u64,
                    );
                    self.bufs[id] = Some(out);
                }
                NodeKind::Move { a, b, kext, pair, a_idx, write_idx, write_base, reduce_max } => {
                    let ar = resolve_read(g, &self.hazard, a);
                    let br = b.as_ref().map(|r| resolve_read(g, &self.hazard, r));
                    if write_idx.is_empty()
                        && b.is_none()
                        && *kext == 1
                        && ar.nls.is_empty()
                        && !self.hazard.contains(&ar.src)
                        && identity_move(g, node, &ar, a_idx, n)
                    {
                        let sb = self.use_buf(ar.src);
                        self.bufs[id] = Some(sb);
                        self.elided += 1;
                    } else if write_idx.is_empty() && b.is_none() && *kext == 1 && a_idx.is_empty() {
                        if self.hazard.contains(&ar.src) {
                            self.emit_plain_copy(id, &ar, n);
                        } else {
                            self.pending.insert(id, (ar, n));
                        }
                    } else if !write_idx.is_empty() {
                        self.emit_scatter(g, id, &ar, write_idx, write_base.unwrap(), &dims);
                    } else if !a_idx.is_empty() {
                        self.emit_gather(g, id, &ar, a_idx, n);
                    } else if *pair && br.is_some() && *kext > 1 {
                        self.emit_contract(g, id, &ar, br.as_ref().unwrap(), *kext, &dims);
                    } else if *pair && br.is_some() {
                        self.emit_binary(g, id, "mul_strided", &ar, br.as_ref().unwrap(), &dims);
                    } else if !*pair && br.is_some() {
                        self.emit_binary(g, id, "add_strided", &ar, br.as_ref().unwrap(), &dims);
                    } else {
                        self.emit_reduce(g, id, &ar, *kext, n, *reduce_max);
                    }
                }
            }
        }
        let mut flush: Vec<usize> = self
            .pending
            .keys()
            .copied()
            .filter(|id| !referenced[*id])
            .collect();
        flush.sort_unstable();
        for id in flush {
            if let Some((r, n)) = self.pending.remove(&id) {
                self.emit_plain_copy(id, &r, n);
            }
        }
        self.elided += self.pending.len();
    }

    fn attn_setup(
        &self,
        g: &GraphIr,
        id: usize,
        mask: usize,
        scale: f32,
        sc: usize,
    ) -> ([i32; 25], bool, usize, usize, usize) {
        let NodeKind::Move { a: qa, b: Some(kb), kext: hd, .. } = &g.nodes[sc].kind else {
            panic!("attn scores node shape")
        };
        let NodeKind::Move { a: _pa, b: Some(vb), .. } = &g.nodes[id].kind else {
            panic!("attn pv node shape")
        };
        let scd = &g.nodes[sc].dims;
        let (kv, grp, s, ctx) = (scd[0], scd[1], scd[2], scd[3]);
        let qa = resolve_read(g, &self.hazard, qa);
        let kb = resolve_read(g, &self.hazard, kb);
        let vb = resolve_read(g, &self.hazard, vb);
        assert!(qa.nls.is_empty() && kb.nls.is_empty() && vb.nls.is_empty(), "attn kernels cannot apply inline nls");
        let (_, qst) = axes_of(&qa.layout);
        let (_, kst) = axes_of(&kb.layout);
        let (_, vst) = axes_of(&vb.layout);
        let md = &g.nodes[mask].dims;
        assert_eq!(md[1], ctx, "mask ctx");
        let mut p = [0i32; 25];
        p[0] = kv as i32;
        p[1] = grp as i32;
        p[2] = s as i32;
        p[3] = ctx as i32;
        p[4] = *hd as i32;
        p[6] = qst[0] as i32;
        p[7] = qst[1] as i32;
        p[8] = qst[2] as i32;
        p[9] = qst[4] as i32;
        p[11] = kst[0] as i32;
        p[12] = kst[3] as i32;
        p[13] = kst[4] as i32;
        p[15] = vst[0] as i32;
        p[16] = vst[4] as i32;
        p[17] = vst[3] as i32;
        p[19] = 0;
        p[20] = 0;
        p[21] = ctx as i32;
        p[22] = 1;
        p[23] = scale.to_bits() as i32;
        let vec_ok = *hd % 4 == 0
            && qst[4] == 1
            && kst[4] == 1
            && [qst[0], qst[1], qst[2], kst[0], kst[3]].iter().all(|st| st % 4 == 0);
        (p, vec_ok, qa.src, kb.src, vb.src)
    }

    fn emit_attn(
        &mut self,
        g: &GraphIr,
        id: usize,
        _q: usize,
        _kc: usize,
        _v: usize,
        mask: usize,
        scale: f32,
        sc: usize,
    ) {
        let (p, vec_ok, qsrc, ksrc, vsrc) = self.attn_setup(g, id, mask, scale, sc);
        let n_out = numel(&g.nodes[id].dims);
        let out = self.alloc(n_out as usize * 4);
        let (qbuf, kbuf, vbuf, mbuf) = (
            self.use_buf(qsrc),
            self.use_buf(ksrc),
            self.use_buf(vsrc),
            self.use_buf(mask),
        );
        self.guard_write_overlap(id, &out, &[&qbuf, &kbuf, &vbuf, &mbuf]);
        let comp = crate::kernelset::Composition::Flash(crate::kernelset::AttnShape { p, vec_ok });
        let (kidx, variant) = if std::env::var("FORCE_MLX_ATTN").is_ok() || std::env::var("FORCE_MLX_ALL").is_ok() {
            (self.backend.kernel_index("mlx_sdpa"), crate::kernelset::Variant([0; 5]))
        } else {
            self.backend.search_cached(&comp)
        };
        self.tape.push(Tape::Kernel {
            kidx,
            variant,
            bufs: vec![qbuf, kbuf, vbuf, mbuf, out.clone()],
            comp,
        });
        self.dispatches += 1;
        self.bufs[id] = Some(out);
    }

    fn emit_reduce_region(&mut self, g: &GraphIr, rr: &crate::coarsen::ReduceRegion) {
        use std::collections::{HashMap, HashSet};
        use std::hash::{Hash, Hasher};
        let internal = rr.internal();
        let s0 = &rr.stages[0];
        let NodeKind::Move { kext, .. } = &g.nodes[s0.reduce].kind else { return };
        let cols = *kext;
        let rows = numel(&g.nodes[s0.reduce].dims);
        let root = rr.root();
        let n_full = numel(&g.nodes[root].dims);

        let mut ext: Vec<(usize, [i32; 18])> = Vec::new();
        let mut slot: HashMap<String, usize> = HashMap::new();
        let mut special: HashMap<usize, String> = HashMap::new();
        let mut stage_src: Vec<(String, bool, String)> = Vec::new();

        for (k, st) in rr.stages.iter().enumerate() {
            let NodeKind::Move { a: r_a, .. } = &g.nodes[st.reduce].kind else { return };
            let r_a = r_a.clone();
            let pre = gen_operand(g, &r_a, &internal, &special, "e", &mut ext, &mut slot);
            let post_tail = st.post.last().copied().unwrap_or(st.reduce);
            let vexpr = if st.post.is_empty() {
                format!("t{k}")
            } else {
                let post_set: HashSet<usize> = st.post.iter().copied().collect();
                let mut sp2 = special.clone();
                sp2.insert(st.reduce, format!("t{k}"));
                gen_node(g, post_tail, &post_set, &sp2, "row", &mut ext, &mut slot)
            };
            stage_src.push((pre, st.reduce_max, vexpr));
            special.insert(st.reduce, format!("t{k}"));
            special.insert(post_tail, format!("v{k}"));
        }
        let out_expr = gen_node(g, root, &internal, &special, "e", &mut ext, &mut slot);

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for (p, m, v) in &stage_src {
            p.hash(&mut hasher);
            m.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        out_expr.hash(&mut hasher);
        ext.len().hash(&mut hasher);
        let h = hasher.finish();
        let name: &'static str = *self
            .fused_names
            .entry(h)
            .or_insert_with(|| Box::leak(format!("rfused_{h:x}").into_boxed_str()));
        self.backend.ensure_reduce_fused(name, ext.len(), &stage_src, &out_expr);

        let out = self.alloc(n_full as usize * 4);
        let mut bufs = vec![out.clone()];
        for (s, _) in &ext {
            bufs.push(self.use_buf(*s));
        }
        let b = 1 + ext.len();
        let scalars = vec![(b as u64, rows as u32), ((b + 1) as u64, cols as u32)];
        let metas: Vec<(u64, [i32; 18])> =
            ext.iter().enumerate().map(|(k, (_, m))| ((b + 2 + k) as u64, *m)).collect();
        self.push1d(name, bufs, metas, scalars, vec![], (rows * 256) as u64);
        self.bufs[root] = Some(out);
    }

    fn emit_fused_region(&mut self, g: &GraphIr, region: &[usize], root: usize) {
        use std::hash::{Hash, Hasher};
        let dims = g.nodes[root].dims.clone();
        let n = numel(&dims);
        let (hdr, body, ext) = gen_region_body(g, region);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hdr.hash(&mut hasher);
        body.hash(&mut hasher);
        root.hash(&mut hasher);
        let h = hasher.finish();
        let name: &'static str = *self
            .fused_names
            .entry(h)
            .or_insert_with(|| Box::leak(format!("fused_{h:x}").into_boxed_str()));
        self.backend.ensure_fused(name, &hdr, &body, root);
        let out = self.alloc(n as usize * 4);
        let mut bufs = vec![out.clone()];
        for (s, _) in &ext {
            bufs.push(self.use_buf(*s));
        }
        let b = 1 + ext.len();
        let scalars = vec![(b as u64, n as u32)];
        let metas: Vec<(u64, [i32; 18])> =
            ext.iter().enumerate().map(|(k, (_, m))| ((b + 1 + k) as u64, *m)).collect();
        self.push1d(name, bufs, metas, scalars, vec![], n as u64);
        self.bufs[root] = Some(out);
    }

    fn emit_binary(&mut self, _g: &GraphIr, id: usize, kernel: &'static str, a: &Read, b: &Read, dims: &[i64]) {
        let n = numel(dims);
        let (sha, sta) = axes_of(&a.layout);
        let (shb, stb) = axes_of(&b.layout);
        let ma = meta_from_axes(&sha, &sta, sha.len() - 1).expect("bin meta a");
        let mb = meta_from_axes(&shb, &stb, shb.len() - 1).expect("bin meta b");
        let out = self.alloc(n as usize * 4);
        let (ab, bb) = (self.use_buf(a.src), self.use_buf(b.src));
        self.guard_write_overlap(id, &out, &[&ab, &bb]);
        self.push1d_off(
            kernel,
            vec![ab, bb, out.clone()],
            vec![(a.base * 4) as u64, (b.base * 4) as u64, 0],
            vec![(4, ma), (5, mb)],
            vec![(3, n as u32), (6, pack_nls(&a.nls)), (7, pack_nls(&b.nls))],
            vec![],
            n as u64,
        );
        self.bufs[id] = Some(out);
    }

    fn emit_reduce(&mut self, _g: &GraphIr, id: usize, a: &Read, kext: i64, n: i64, reduce_max: bool) {
        let meta = meta_of(&a.layout).expect("reduce meta");
        let out = self.alloc(n as usize * 4);
        let sb = self.use_buf(a.src);
        self.guard_write_overlap(id, &out, &[&sb]);
        let kernel = if reduce_max { "reduce_max_l" } else { "reduce_sum_l" };
        self.push1d_off(
            kernel,
            vec![sb, out.clone()],
            vec![(a.base * 4) as u64, 0],
            vec![(5, meta)],
            vec![(2, n as u32), (3, kext as u32), (4, 1u32), (6, pack_nls(&a.nls))],
            vec![],
            n as u64,
        );
        self.bufs[id] = Some(out);
    }

    fn emit_gather(&mut self, _g: &GraphIr, id: usize, a: &Read, idx: &[IdxRead], n: i64) {
        assert!(a.nls.is_empty(), "gather cannot apply inline nls");
        let rmeta = meta_of(&a.layout).expect("gather rmeta");
        let mut rspec = [0i32; 8];
        rspec[0] = idx.len() as i32;
        let mut ibufs = Vec::new();
        for (k, ir) in idx.iter().enumerate() {
            rspec[1 + 2 * k] = k as i32;
            rspec[2 + 2 * k] = ir.stride as i32;
            ibufs.push((self.use_buf(ir.idx), meta_of(&ir.idx_layout).expect("idx meta")));
        }
        let out = self.alloc(n as usize * 4);
        let sb = self.use_buf(a.src);
        {
            let mut reads: Vec<&Buffer> = vec![&sb];
            reads.extend(ibufs.iter().map(|(b, _)| b));
            self.guard_write_overlap(id, &out, &reads);
        }
        self.xcopy(sb, out.clone(), ibufs, n, rmeta, flat_meta(n), rspec, [0; 8], a.base);
        self.bufs[id] = Some(out);
    }

    fn emit_scatter(
        &mut self,
        g: &GraphIr,
        id: usize,
        a: &Read,
        widx: &[IdxRead],
        base: usize,
        dims: &[i64],
    ) {
        assert!(a.nls.is_empty(), "scatter cannot apply inline nls");
        let n = numel(dims);
        let alias = self.inplace_cache && matches!(g.nodes[base].kind, NodeKind::Input { .. });
        let out = if alias {
            self.use_buf(base)
        } else {
            self.alloc(n as usize * 4)
        };
        if !alias {
            let bb = self.use_buf(base);
            self.xcopy(bb, out.clone(), vec![], n, flat_meta(n), flat_meta(n), [0; 8], [0; 8], 0);
        }
        let rmeta = meta_of(&a.layout).expect("scatter rmeta");
        let (wsh, _) = axes_of(&a.layout);
        let src_n: i64 = wsh.iter().product();
        let mut wst = vec![0i64; wsh.len()];
        if wsh.len() >= 2 {
            wst[wsh.len() - 2] = 1;
        }
        let wmeta = meta_from_axes(&wsh, &wst, wsh.len()).expect("scatter wmeta");
        let mut wspec = [0i32; 8];
        wspec[0] = widx.len() as i32;
        let mut ibufs = Vec::new();
        for (k, ir) in widx.iter().enumerate() {
            wspec[1 + 2 * k] = k as i32;
            wspec[2 + 2 * k] = ir.stride as i32;
            ibufs.push((self.use_buf(ir.idx), meta_of(&ir.idx_layout).expect("widx meta")));
        }
        let sb = self.use_buf(a.src);
        {
            let mut reads: Vec<&Buffer> = vec![&sb];
            reads.extend(ibufs.iter().map(|(b, _)| b));
            self.guard_write_overlap(id, &out, &reads);
        }
        self.xcopy(sb, out.clone(), ibufs, src_n, rmeta, wmeta, [0; 8], wspec, a.base);
        self.bufs[id] = Some(out);
    }

    fn emit_contract(&mut self, _g: &GraphIr, id: usize, a: &Read, b: &Read, kext: i64, dims: &[i64]) {
        assert!(a.base == 0 && b.base == 0, "contraction operand base offsets not yet threaded through the Mm path");
        assert!(a.nls.is_empty() && b.nls.is_empty(), "contraction kernels cannot apply inline nls");
        let cl = classify_contract(a, b, kext, dims);
        let (m, nn, ap, bp, cp, batch, nbatch) =
            (cl.m, cl.n, cl.ap, cl.bp, cl.cp, cl.batch.clone(), cl.nbatch);
        let n_out = numel(dims);
        let out = self.alloc(n_out as usize * 4);
        let (ab, bb) = (self.use_buf(a.src), self.use_buf(b.src));
        self.guard_write_overlap(id, &out, &[&ab, &bb]);
        let _ = &batch;
        let comp = crate::kernelset::Composition::Contraction { shape: cl.clone(), kext };
        let mlx_ok = nbatch == 1
            && ap[1] == kext as i32 && ap[2] == 1
            && bp[1] == nn as i32 && bp[2] == 1
            && cp[1] == nn as i32 && cp[2] == 1;
        let (kidx, variant) = if std::env::var("GENERAL_ONLY").is_ok() {
            let name = if nbatch > 1 { "bmm_scalar" } else { "scalar_matmul" };
            (self.backend.kernel_index(name), crate::kernelset::Variant([254, 0, 0, 0, 0]))
        } else if mlx_ok && (std::env::var("FORCE_MLX_MATMUL").is_ok() || std::env::var("FORCE_MLX_ALL").is_ok()) {
            (self.backend.kernel_index("mlx_matmul"), crate::kernelset::Variant([255, 0, 0, 0, 0]))
        } else {
            self.backend.search_cached(&comp)
        };
        self.tape.push(Tape::Kernel {
            kidx,
            variant,
            bufs: vec![ab, bb, out.clone()],
            comp,
        });
        self.dispatches += 1;
        self.bufs[id] = Some(out);
    }

    pub fn dump_tape(&self) {
        use std::collections::BTreeMap;
        let mut c: BTreeMap<&str, usize> = BTreeMap::new();
        for t in &self.tape {
            match t {
                Tape::Run1d { kernel, .. } => *c.entry(kernel).or_default() += 1,
                Tape::Kernel { kidx, .. } => *c.entry(self.backend.kernel_name(*kidx)).or_default() += 1,
            }
        }
        eprintln!("tape: {} entries", self.tape.len());
        for (k, v) in &c {
            eprintln!("  {k} x{v}");
        }
    }

    pub fn replay(&self) {
        let mut cb = self.backend.owned_command_buffer();
        for t in &self.tape {
            match t {
                Tape::Run1d { kernel, bufs, offsets, metas, scalars, specs, progs, n } => {
                    let enc = self.backend.enc(&cb, kernel);
                    enc.set_compute_pipeline_state(self.backend.pipeline(kernel));
                    for (k, b) in bufs.iter().enumerate() {
                        enc.set_buffer(k as u64, Some(b), offsets.get(k).copied().unwrap_or(0));
                    }
                    for (slot, m) in metas {
                        enc.set_bytes(*slot, 72, m.as_ptr() as *const _);
                    }
                    for (slot, v) in scalars {
                        enc.set_bytes(*slot, 4, v as *const u32 as *const _);
                    }
                    for (slot, sp) in specs {
                        enc.set_bytes(*slot, 32, sp.as_ptr() as *const _);
                    }
                    for (slot, pr) in progs {
                        enc.set_bytes(*slot, 100, pr.as_ptr() as *const _);
                    }
                    let tg = 256u64.min((*n).max(1));
                    enc.dispatch_threads(
                        metal::MTLSize::new((*n).max(1), 1, 1),
                        metal::MTLSize::new(tg, 1, 1),
                    );
                    enc.end_encoding();
                }
                Tape::Kernel { kidx, variant, bufs, comp } => {
                    if self.backend.kernel_breaks_cb(*kidx) {
                        cb.commit();
                        cb.wait_until_completed();
                    }
                    self.backend.run_kernel(&cb, *kidx, *variant, bufs, comp);
                    if self.backend.kernel_breaks_cb(*kidx) {
                        cb = self.backend.owned_command_buffer();
                    }
                }
            }
        }
        cb.commit();
        cb.wait_until_completed();
    }

    pub fn read_f32(&self, g: &GraphIr, id: usize) -> Vec<f32> {
        let n = numel(&g.nodes[id].dims) as usize;
        self.backend.read(&self.buf(id), n)
    }

    pub fn first_divergence(&self, g: &GraphIr, refm: &crate::cpu_ref::CpuRef) -> Option<(usize, f32)> {
        for id in 0..g.nodes.len() {
            let Some(buf) = self.bufs[id].as_ref() else { continue };
            if g.nodes[id].dtype != Dt::F32 {
                continue;
            }
            let Some(rv) = refm.f[id].as_ref() else { continue };
            let n = numel(&g.nodes[id].dims) as usize;
            let nv = self.backend.read(buf, n);
            if nv.len() != rv.len() {
                return Some((id, f32::INFINITY));
            }
            let peak = rv.iter().fold(1e-9f32, |m, &x| m.max(x.abs()));
            let maxd = nv.iter().zip(rv).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            if maxd > 1e-3 * peak {
                return Some((id, maxd / peak));
            }
        }
        None
    }

    pub fn first_nan(&self, g: &GraphIr) -> Option<(usize, usize)> {
        for id in 0..g.nodes.len() {
            let Some(buf) = self.bufs[id].as_ref() else { continue };
            if g.nodes[id].dtype != Dt::F32 {
                continue;
            }
            let n = numel(&g.nodes[id].dims) as usize;
            let v = self.backend.read(buf, n);
            if let Some(k) = v.iter().position(|x| x.is_nan()) {
                return Some((id, k));
            }
        }
        None
    }
}
