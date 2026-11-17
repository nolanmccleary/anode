use crate::ir::{Dt, GraphIr, IdxRead, Nl, NodeKind, Read};
use crate::native_fuse;
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

enum Tape {
    Run1d {
        kernel: &'static str,
        bufs: Vec<Buffer>,
        metas: Vec<(u64, [i32; 18])>,
        scalars: Vec<(u64, u32)>,
        specs: Vec<(u64, [i32; 8])>,
        progs: Vec<(u64, [i32; 25])>,
        n: u64,
    },
    Mm {
        a: Buffer,
        b: Buffer,
        c: Buffer,
        m: usize,
        k: usize,
        n: usize,
        ap: [i32; 4],
        bp: [i32; 4],
        cp: [i32; 4],
        batch: Vec<i32>,
        nbatch: i64,
    },
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
        cur = Read { src: a.src, layout: comp };
    }
    cur
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

    fn emit_plain_copy(&mut self, id: usize, a: &Read, n: i64) {
        let meta = meta_of(&a.layout).expect("copy meta");
        let out = self.alloc(n as usize * 4);
        let sb = self.use_buf(a.src);
        self.xcopy(sb, out.clone(), vec![], n, meta, flat_meta(n), [0; 8], [0; 8]);
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
        self.tape.push(Tape::Run1d { kernel, bufs, metas, scalars, specs, progs: Vec::new(), n });
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
    ) {
        let d = |k: usize, idx: &Vec<(Buffer, [i32; 18])>| {
            idx.get(k).map(|(b, _)| b.clone()).unwrap_or_else(|| src.clone())
        };
        let (i0, i1, i2, i3) = (d(0, &idx), d(1, &idx), d(2, &idx), d(3, &idx));
        let im = |k: usize| idx.get(k).map(|(_, m)| *m).unwrap_or([0i32; 18]);
        self.push1d(
            "xcopy_f",
            vec![src, dst, i0, i1, i2, i3],
            vec![(7, rmeta), (8, wmeta), (11, im(0)), (12, im(1)), (13, im(2)), (14, im(3))],
            vec![(6, n as u32)],
            vec![(9, rspec), (10, wspec)],
            n as u64,
        );
    }

    pub fn build(&mut self, g: &GraphIr, inputs: &std::collections::HashMap<String, Vec<f32>>) {
        self.bufs = vec![None; g.nodes.len()];
        let fusions = if std::env::var("GENERAL_ONLY").is_ok() {
            native_fuse::Fusions {
                claims: Default::default(),
                rms: Default::default(),
                silu: Default::default(),
                attn: Default::default(),
            }
        } else {
            native_fuse::plan(g)
        };
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
        let litf = |id: usize| -> f32 {
            match &g.nodes[id].kind {
                NodeKind::LiteralF32(v) => v[0],
                _ => panic!("expected scalar literal"),
            }
        };
        for (id, node) in g.nodes.iter().enumerate() {
            if fusions.claims.contains(&id) {
                continue;
            }
            if let Some((x, ce)) = fusions.rms.get(&id) {
                let xd = g.nodes[*x].dims.clone();
                let cols = *xd.last().unwrap();
                let rows = numel(&xd) / cols;
                let out = self.alloc(numel(&xd) as usize * 4);
                let xb = self.use_buf(*x);
                self.push1d(
                    "rmsnorm_row",
                    vec![xb, out.clone()],
                    vec![],
                    vec![(2, rows as u32), (3, cols as u32), (4, litf(*ce).to_bits())],
                    vec![],
                    rows as u64 * 256,
                );
                self.bufs[id] = Some(out);
                continue;
            }
            if let Some((gg, up, cl, c1)) = fusions.silu.get(&id) {
                let n = numel(&g.nodes[id].dims);
                let out = self.alloc(n as usize * 4);
                let (gb, ub) = (self.use_buf(*gg), self.use_buf(*up));
                self.push1d(
                    "silu_mul",
                    vec![gb, ub, out.clone()],
                    vec![],
                    vec![(3, n as u32), (4, litf(*cl).to_bits()), (5, litf(*c1).to_bits())],
                    vec![],
                    n as u64,
                );
                self.bufs[id] = Some(out);
                continue;
            }
            if let Some((q, kc, v, mask, cs, sc)) = fusions.attn.get(&id) {
                self.emit_attn(g, id, *q, *kc, *v, *mask, litf(*cs), *sc);
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
                    let meta = meta_of(&rr.layout).expect("pointwise meta");
                    let out = self.alloc(n as usize * 4);
                    let sb = self.use_buf(rr.src);
                    self.push1d(
                        kernel,
                        vec![sb, out.clone()],
                        vec![(3, meta)],
                        vec![(2, n as u32)],
                        vec![],
                        n as u64,
                    );
                    self.bufs[id] = Some(out);
                }
                NodeKind::Move { a, b, kext, pair, a_idx, write_idx, write_base } => {
                    let ar = resolve_read(g, &self.hazard, a);
                    let br = b.as_ref().map(|r| resolve_read(g, &self.hazard, r));
                    if write_idx.is_empty()
                        && b.is_none()
                        && *kext == 1
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
                        self.emit_reduce(g, id, &ar, *kext, n);
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
        let NodeKind::Move { a: qa, b: Some(kb), kext: hd, .. } = &g.nodes[sc].kind else {
            panic!("attn scores node shape")
        };
        let NodeKind::Move { a: _pa, b: Some(vb), .. } = &g.nodes[id].kind else {
            panic!("attn pv node shape")
        };
        let scd = &g.nodes[sc].dims;
        let (kv, grp, s, ctx) = (scd[0], scd[1], scd[2], scd[3]);
        let qa = &resolve_read(g, &self.hazard, qa);
        let kb = &resolve_read(g, &self.hazard, kb);
        let vb = &resolve_read(g, &self.hazard, vb);
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
        let (split, vec) = self.backend.tune_attn(&p, vec_ok);
        p[24] = (split | (vec << 8)) as i32;
        let n_out = numel(&g.nodes[id].dims);
        let out = self.alloc(n_out as usize * 4);
        let (qbuf, kbuf, vbuf, mbuf) = (
            self.use_buf(qa.src),
            self.use_buf(kb.src),
            self.use_buf(vb.src),
            self.use_buf(mask),
        );
        let jobs = (kv * grp * s) as u64;
        let threads = jobs.div_ceil(8 / split as u64) * 256;
        self.tape.push(Tape::Run1d {
            kernel: "attn_online",
            bufs: vec![qbuf, kbuf, vbuf, mbuf, out.clone()],
            metas: vec![],
            scalars: vec![],
            specs: vec![],
            progs: vec![(5, p)],
            n: threads,
        });
        self.dispatches += 1;
        self.bufs[id] = Some(out);
    }

    fn emit_binary(&mut self, _g: &GraphIr, id: usize, kernel: &'static str, a: &Read, b: &Read, dims: &[i64]) {
        let n = numel(dims);
        let (sha, sta) = axes_of(&a.layout);
        let (shb, stb) = axes_of(&b.layout);
        let ma = meta_from_axes(&sha, &sta, sha.len() - 1).expect("bin meta a");
        let mb = meta_from_axes(&shb, &stb, shb.len() - 1).expect("bin meta b");
        let out = self.alloc(n as usize * 4);
        let (ab, bb) = (self.use_buf(a.src), self.use_buf(b.src));
        self.push1d(
            kernel,
            vec![ab, bb, out.clone()],
            vec![(4, ma), (5, mb)],
            vec![(3, n as u32)],
            vec![],
            n as u64,
        );
        self.bufs[id] = Some(out);
    }

    fn emit_reduce(&mut self, _g: &GraphIr, id: usize, a: &Read, kext: i64, n: i64) {
        let meta = meta_of(&a.layout).expect("reduce meta");
        let out = self.alloc(n as usize * 4);
        let sb = self.use_buf(a.src);
        self.push1d(
            "reduce_sum_l",
            vec![sb, out.clone()],
            vec![(5, meta)],
            vec![(2, n as u32), (3, kext as u32), (4, 1u32)],
            vec![],
            n as u64,
        );
        self.bufs[id] = Some(out);
    }

    fn emit_gather(&mut self, _g: &GraphIr, id: usize, a: &Read, idx: &[IdxRead], n: i64) {
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
        self.xcopy(sb, out.clone(), ibufs, n, rmeta, flat_meta(n), rspec, [0; 8]);
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
        let n = numel(dims);
        let alias = self.inplace_cache && matches!(g.nodes[base].kind, NodeKind::Input { .. });
        let out = if alias {
            self.use_buf(base)
        } else {
            self.alloc(n as usize * 4)
        };
        if !alias {
            let bb = self.use_buf(base);
            self.xcopy(bb, out.clone(), vec![], n, flat_meta(n), flat_meta(n), [0; 8], [0; 8]);
        }
        let src_n = numel(&g.nodes[a.src].dims);
        let rmeta = meta_of(&a.layout).expect("scatter rmeta");
        let (wsh, _) = axes_of(&a.layout);
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
        self.xcopy(sb, out.clone(), ibufs, src_n, rmeta, wmeta, [0; 8], wspec);
        self.bufs[id] = Some(out);
    }

    fn emit_contract(&mut self, _g: &GraphIr, id: usize, a: &Read, b: &Read, kext: i64, dims: &[i64]) {
        let (sha, sta) = axes_of(&a.layout);
        let (_shb, stb) = axes_of(&b.layout);
        let rank = sha.len();
        let kax = rank - 1;
        assert_eq!(sha[kax], kext, "k axis extent");
        let mut m_ax = None;
        let mut n_ax = None;
        let mut b_axes: Vec<usize> = Vec::new();
        for ax in 0..kax {
            if dims[ax] == 1 {
                continue;
            }
            match (sta[ax] != 0, stb[ax] != 0) {
                (true, false) => {
                    if m_ax.is_none() {
                        m_ax = Some(ax);
                    } else {
                        b_axes.push(ax);
                    }
                }
                (false, true) => {
                    if n_ax.is_none() {
                        n_ax = Some(ax);
                    } else {
                        b_axes.push(ax);
                    }
                }
                _ => b_axes.push(ax),
            }
        }
        for ax in 0..kax {
            if dims[ax] == 1 {
                continue;
            }
            if m_ax.is_none() && !b_axes.contains(&ax) && Some(ax) != n_ax {
                m_ax = Some(ax);
            }
        }
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
        let n_out = numel(dims);
        assert_eq!(n_out, m * nn * nbatch, "contract extents");
        let ap = [0, am_s as i32, sta[kax] as i32, 0];
        let bp = [0, stb[kax] as i32, bn_s as i32, 0];
        let crs = if m_ax.is_some() { cm_s } else { 1 };
        let ccs = if n_ax.is_some() { cn_s } else { 1 };
        let cp = [0, crs as i32, ccs as i32, 0];
        let out = self.alloc(n_out as usize * 4);
        let (ab, bb) = (self.use_buf(a.src), self.use_buf(b.src));
        if nbatch == 1 && m == 1 && (bp[2] == 1 || bp[1] == 1) {
            self.backend
                .tune_gemv(kext as usize, nn as usize, &ab, &bb, ap, bp);
        }
        if nbatch == 1 && m > 1 && std::env::var("GENERAL_ONLY").is_err() {
            self.backend
                .tune_gemm(m as usize, kext as usize, nn as usize, ap, bp, cp);
        }
        self.tape.push(Tape::Mm {
            a: ab,
            b: bb,
            c: out.clone(),
            m: m as usize,
            k: kext as usize,
            n: nn as usize,
            ap,
            bp,
            cp,
            batch,
            nbatch,
        });
        self.dispatches += 1;
        self.bufs[id] = Some(out);
    }

    pub fn replay(&self) {
        let cb = self.backend.owned_command_buffer();
        for t in &self.tape {
            match t {
                Tape::Run1d { kernel, bufs, metas, scalars, specs, progs, n } => {
                    let enc = cb.new_compute_command_encoder();
                    if *kernel == "attn_online" {
                        let pr = progs[0].1;
                        let split = (pr[24] & 0xff).max(1) as u32;
                        let vec = ((pr[24] >> 8) & 0xff).max(1) as u32;
                        let pipe = self.backend.attn_pipeline(split, vec);
                        enc.set_compute_pipeline_state(&pipe);
                    } else {
                        enc.set_compute_pipeline_state(self.backend.pipeline(kernel));
                    }
                    for (k, b) in bufs.iter().enumerate() {
                        enc.set_buffer(k as u64, Some(b), 0);
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
                Tape::Mm { a, b, c, m, k, n, ap, bp, cp, batch, nbatch } => {
                    if *nbatch > 1 {
                        self.backend
                            .bmm_into(&cb, a, b, c, *m, *k, *n, *ap, *bp, *cp, batch, *nbatch as u64);
                    } else {
                        self.backend
                            .matmul_or_gemv_into(&cb, a, b, c, *m, *k, *n, *ap, *bp, *cp);
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
}
