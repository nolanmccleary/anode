use crate::metal_backend::CuteMetal;
use crate::native_exec::ContractShape;
use metal::{Buffer, CommandBufferRef};

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ServicerTag {
    Contraction,
    Flash,
}

#[derive(Clone)]
pub struct AttnShape {
    pub p: [i32; 25],
    pub vec_ok: bool,
}

#[derive(Clone)]
pub enum Composition {
    Contraction { shape: ContractShape, kext: i64 },
    Flash(AttnShape),
}

impl Composition {
    pub fn tag(&self) -> ServicerTag {
        match self {
            Composition::Contraction { .. } => ServicerTag::Contraction,
            Composition::Flash(_) => ServicerTag::Flash,
        }
    }

    pub fn key(&self) -> Vec<i64> {
        match self {
            Composition::Contraction { shape: s, kext } => vec![
                0, s.m as i64, *kext, s.n as i64, s.nbatch,
                s.ap[1] as i64, s.ap[2] as i64, s.bp[1] as i64, s.bp[2] as i64,
                s.cp[1] as i64, s.cp[2] as i64, (s.bp[0] % 4) as i64, (s.ap[0] % 4) as i64,
            ],
            Composition::Flash(a) => vec![
                1, (a.p[0] * a.p[1] * a.p[2]) as i64, a.p[3] as i64, a.p[4] as i64, a.vec_ok as i64,
            ],
        }
    }

    pub fn contraction(&self) -> &ContractShape {
        match self {
            Composition::Contraction { shape, .. } => shape,
            _ => unreachable!(),
        }
    }

    pub fn kext(&self) -> i64 {
        match self {
            Composition::Contraction { kext, .. } => *kext,
            _ => unreachable!(),
        }
    }

    pub fn flash(&self) -> &AttnShape {
        match self {
            Composition::Flash(a) => a,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Variant(pub [u32; 5]);

pub trait Kernel {
    fn name(&self) -> &'static str;
    fn servicer(&self) -> ServicerTag;
    fn admits(&self, c: &Composition) -> bool;
    fn variants(&self, c: &Composition) -> Vec<Variant>;
    fn breaks_cb(&self) -> bool {
        false
    }
    fn floor(&self) -> bool {
        false
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], v: Variant, c: &Composition);
}

pub trait Cost {
    fn rank(&self, b: &CuteMetal, k: &dyn Kernel, v: Variant, c: &Composition, bufs: &[Buffer]) -> f64;
}

fn foot(sh: &[i64], st: &[i64]) -> usize {
    let mut f = 1usize;
    for (s, d) in sh.iter().zip(st) {
        f += (*s - 1).max(0) as usize * d.unsigned_abs() as usize;
    }
    (f + 3) & !3
}

fn bench_bufs(b: &CuteMetal, c: &Composition) -> (Vec<Buffer>, Composition) {
    match c {
        Composition::Contraction { shape, kext } => {
            let (m, n, k) = (shape.m as i64, shape.n as i64, *kext);
            let afoot = foot(&[m, k], &[shape.ap[1] as i64, shape.ap[2] as i64]);
            let bfoot = foot(&[k, n], &[shape.bp[1] as i64, shape.bp[2] as i64]);
            let cfoot = foot(&[m, n], &[shape.cp[1] as i64, shape.cp[2] as i64]);
            let bufs = vec![
                b.alloc_bytes(afoot * 4),
                b.alloc_bytes(bfoot * 4),
                b.alloc_bytes(cfoot * 4),
            ];
            let z = ContractShape {
                m: shape.m,
                n: shape.n,
                ap: [0, shape.ap[1], shape.ap[2], 0],
                bp: [0, shape.bp[1], shape.bp[2], 0],
                cp: [0, shape.cp[1], shape.cp[2], 0],
                batch: shape.batch.clone(),
                nbatch: shape.nbatch,
            };
            (bufs, Composition::Contraction { shape: z, kext: *kext })
        }
        Composition::Flash(a) => {
            let p = a.p;
            let (kv, g, s, ctx, hd) = (p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize, p[4] as usize);
            let jobs = kv * g * s;
            let af = |s0: i32, n0: usize, s1: i32, n1: usize, s2: i32, n2: usize, s3: i32, n3: usize| -> usize {
                1 + s0.max(0) as usize * (n0 - 1)
                    + s1.max(0) as usize * (n1 - 1)
                    + s2.max(0) as usize * (n2 - 1)
                    + s3.max(0) as usize * (n3 - 1)
            };
            let kfoot = (af(p[11], kv, p[12], ctx, p[13], hd, 0, 1) + 3) & !3;
            let vfoot = (af(p[15], kv, p[16], ctx, p[17], hd, 0, 1) + 3) & !3;
            let qfoot = af(p[6], kv, p[7], g, p[8], s, p[9], hd);
            let bfoot = af(p[19], kv, p[20], g, p[21], s, p[22], ctx);
            let bufs = vec![
                b.alloc_bytes(qfoot * 4),
                b.alloc_bytes(kfoot * 4),
                b.alloc_bytes(vfoot * 4),
                b.alloc_bytes(bfoot * 4),
                b.alloc_bytes(jobs * hd * 4),
            ];
            let mut pz = p;
            pz[5] = 0;
            pz[10] = 0;
            pz[14] = 0;
            pz[18] = 0;
            (bufs, Composition::Flash(AttnShape { p: pz, vec_ok: a.vec_ok }))
        }
    }
}

pub struct IsolatedBench;

impl Cost for IsolatedBench {
    fn rank(&self, b: &CuteMetal, k: &dyn Kernel, v: Variant, c: &Composition, bufs: &[Buffer]) -> f64 {
        let run = |iters: usize| {
            let cb = b.owned_command_buffer();
            for _ in 0..iters {
                k.dispatch(b, &cb, bufs, v, c);
            }
            cb.commit();
            cb.wait_until_completed();
        };
        run(1);
        let t = std::time::Instant::now();
        run(2);
        t.elapsed().as_secs_f64() * 1e3 / 2.0
    }
}

pub fn search(b: &CuteMetal, kernels: &[Box<dyn Kernel>], cost: &dyn Cost, c: &Composition) -> Option<(usize, Variant)> {
    let tag = c.tag();
    let any = kernels.iter().any(|k| k.servicer() == tag && k.admits(c));
    if !any {
        return None;
    }
    let non_floor = kernels.iter().any(|k| k.servicer() == tag && k.admits(c) && !k.floor());
    let (bufs, zc) = bench_bufs(b, c);
    let mut best: Option<(f64, usize, Variant)> = None;
    for (ki, k) in kernels.iter().enumerate() {
        if k.servicer() != tag || !k.admits(&zc) {
            continue;
        }
        if non_floor && k.floor() {
            continue;
        }
        for v in k.variants(&zc) {
            let ms = cost.rank(b, k.as_ref(), v, &zc, &bufs);
            if best.map(|(bm, _, _)| ms < bm).unwrap_or(true) {
                best = Some((ms, ki, v));
            }
        }
    }
    best.map(|(_, ki, v)| (ki, v))
}

fn col(c: &Composition) -> bool {
    c.contraction().m == 1 && c.contraction().bp[2] == 1
}

fn rows(c: &Composition) -> bool {
    c.contraction().m == 1 && c.contraction().bp[1] == 1 && c.contraction().bp[2] != 1
}

fn col_vec4(c: &Composition) -> bool {
    let bp = c.contraction().bp;
    bp[0] % 4 == 0 && bp[1] % 4 == 0
}

fn gemv_io<'a>(c: &Composition, bufs: &'a [Buffer]) -> (&'a Buffer, &'a Buffer, &'a Buffer, i64, i64, [i32; 4], [i32; 4], i32) {
    let s = c.contraction();
    let (k, n) = (c.kext(), s.n as i64);
    let xap = [s.ap[0], s.ap[2], 0, 0];
    let wbp = [s.bp[0], s.bp[1], 0, 0];
    (&bufs[0], &bufs[1], &bufs[2], k, n, xap, wbp, s.cp[0])
}

pub struct GemvT;
impl Kernel for GemvT {
    fn name(&self) -> &'static str {
        "gemv_t"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Contraction
    }
    fn admits(&self, c: &Composition) -> bool {
        col(c)
    }
    fn variants(&self, c: &Composition) -> Vec<Variant> {
        let vecs: &[u32] = if col_vec4(c) { &[4, 8] } else { &[4] };
        let mut v = Vec::new();
        for tg in [128u32, 256u32] {
            for ncolg in [4u32, 8u32, 16u32] {
                let ks = tg / ncolg;
                if !(8..=64).contains(&ks) {
                    continue;
                }
                for vec in vecs {
                    v.push(Variant([0, *vec, ncolg, tg, 0]));
                }
            }
        }
        v
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], v: Variant, c: &Composition) {
        let (a, w, out, k, n, xap, wbp, co) = gemv_io(c, bufs);
        let (vec, ncolg, tg) = (v.0[1], v.0[2], v.0[3] as u64);
        let pipe = b.gemv_pipeline(vec, ncolg);
        b.encode_gemv_t(cb, &pipe, a, w, out, k as usize, n as usize, xap, wbp, co, vec, ncolg, tg);
    }
}

pub struct GemvSg;
impl Kernel for GemvSg {
    fn name(&self) -> &'static str {
        "gemv_sg"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Contraction
    }
    fn admits(&self, c: &Composition) -> bool {
        col(c)
    }
    fn variants(&self, _c: &Composition) -> Vec<Variant> {
        let mut v = Vec::new();
        for tg in [128u32, 256u32] {
            for csplit in [4u32, 8u32, 16u32] {
                v.push(Variant([1, 4, csplit, tg, 0]));
            }
        }
        v
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], v: Variant, c: &Composition) {
        let (a, w, out, k, n, xap, wbp, co) = gemv_io(c, bufs);
        let (csplit, tg) = (v.0[2], v.0[3] as u64);
        let pipe = b.gemv_sg_pipeline(csplit);
        b.encode_gemv_sg(cb, &pipe, a, w, out, k as usize, n as usize, xap, wbp, co, csplit, tg);
    }
}

pub struct GemvMlx;
impl Kernel for GemvMlx {
    fn name(&self) -> &'static str {
        "gemv_mlx"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Contraction
    }
    fn admits(&self, c: &Composition) -> bool {
        col(c) && col_vec4(c)
    }
    fn variants(&self, _c: &Composition) -> Vec<Variant> {
        [32u32, 64u32, 128u32].iter().map(|tg| Variant([3, 4, 1, *tg, 0])).collect()
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], v: Variant, c: &Composition) {
        let (a, w, out, k, n, xap, wbp, co) = gemv_io(c, bufs);
        b.encode_gemv_mlx(cb, a, w, out, k as usize, n as usize, xap, wbp, co, v.0[3] as u64);
    }
}

pub struct GemvRows;
impl Kernel for GemvRows {
    fn name(&self) -> &'static str {
        "gemv_rows"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Contraction
    }
    fn admits(&self, c: &Composition) -> bool {
        rows(c)
    }
    fn variants(&self, c: &Composition) -> Vec<Variant> {
        let s = c.contraction();
        let rv = if s.bp[0] % 4 == 0 && s.bp[2] % 4 == 0 && c.kext() % 4 == 0 && s.ap[2] == 1 { 4 } else { 1 };
        [128u32, 256u32].iter().map(|tg| Variant([2, rv, 1, *tg, 0])).collect()
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], v: Variant, c: &Composition) {
        let s = c.contraction();
        let (a, w, out) = (&bufs[0], &bufs[1], &bufs[2]);
        let (k, n) = (c.kext(), s.n as i64);
        let xap = [s.ap[0], s.ap[2], 0, 0];
        let wbp = [s.bp[0], 1, s.bp[2], 0];
        let (vec, tg) = (v.0[1], v.0[3] as u64);
        let pipe = b.gemv_rows_pipeline(vec);
        b.encode_gemv_rows(cb, &pipe, a, w, out, k as usize, n as usize, xap, wbp, s.cp[0], tg);
    }
}

fn gemm_ok(c: &Composition) -> bool {
    c.contraction().m > 1
}

fn scalar_ok(c: &Composition) -> bool {
    let s = c.contraction();
    s.m > 1 && s.nbatch == 1
}

fn bmm_scalar_ok(c: &Composition) -> bool {
    let s = c.contraction();
    s.m > 1 && s.nbatch > 1
}

fn gemm_sg_ok(c: &Composition) -> bool {
    if !gemm_ok(c) {
        return false;
    }
    let s = c.contraction();
    let k = c.kext() as usize;
    let ap0 = [0, s.ap[1], s.ap[2], 0];
    let bp0 = [0, s.bp[1], s.bp[2], 0];
    let cp0 = [0, s.cp[1], s.cp[2], 0];
    crate::atoms::mma_plan(&crate::atoms::SG_F32_8X8X8, s.m, k, s.n, ap0, bp0, cp0).is_some()
        && !crate::atoms::mma_candidates(s.m, s.n, k).is_empty()
}

fn gemm_mlx_ok(c: &Composition) -> bool {
    let s = c.contraction();
    scalar_ok(c) && s.m % 8 == 0 && s.n % 8 == 0 && c.kext() % 16 == 0 && s.ap[2] == 1 && s.bp[2] == 1
}

#[allow(clippy::type_complexity)]
fn gemm_io<'a>(c: &'a Composition, bufs: &'a [Buffer]) -> (&'a Buffer, &'a Buffer, &'a Buffer, usize, usize, usize, [i32; 4], [i32; 4], [i32; 4], &'a [i32], i64) {
    let s = c.contraction();
    (&bufs[0], &bufs[1], &bufs[2], s.m, c.kext() as usize, s.n, s.ap, s.bp, s.cp, &s.batch, s.nbatch)
}

pub struct ScalarMatmul;
impl Kernel for ScalarMatmul {
    fn name(&self) -> &'static str {
        "scalar_matmul"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Contraction
    }
    fn admits(&self, c: &Composition) -> bool {
        scalar_ok(c)
    }
    fn floor(&self) -> bool {
        true
    }
    fn variants(&self, _c: &Composition) -> Vec<Variant> {
        vec![Variant([254, 0, 0, 0, 0])]
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], _v: Variant, c: &Composition) {
        let (a, w, o, m, k, n, ap, bp, cp, _batch, _nbatch) = gemm_io(c, bufs);
        b.matmul_into(cb, a, w, o, m, k, n, ap, bp, cp);
    }
}

pub struct BmmScalar;
impl Kernel for BmmScalar {
    fn name(&self) -> &'static str {
        "bmm_scalar"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Contraction
    }
    fn admits(&self, c: &Composition) -> bool {
        bmm_scalar_ok(c)
    }
    fn floor(&self) -> bool {
        true
    }
    fn variants(&self, _c: &Composition) -> Vec<Variant> {
        vec![Variant([254, 0, 0, 0, 0])]
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], _v: Variant, c: &Composition) {
        let (a, w, o, m, k, n, ap, bp, cp, batch, nbatch) = gemm_io(c, bufs);
        b.bmm_scalar(cb, a, w, o, m, k, n, ap, bp, cp, batch, nbatch as u64);
    }
}

pub struct GemmSg;
impl Kernel for GemmSg {
    fn name(&self) -> &'static str {
        "gemm_sg"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Contraction
    }
    fn admits(&self, c: &Composition) -> bool {
        gemm_sg_ok(c)
    }
    fn variants(&self, c: &Composition) -> Vec<Variant> {
        let s = c.contraction();
        crate::atoms::mma_candidates(s.m, s.n, c.kext() as usize)
            .into_iter()
            .map(|t| Variant([t.0, t.1, t.2, t.3, t.4]))
            .collect()
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], v: Variant, c: &Composition) {
        let (a, w, o, m, k, n, ap, bp, cp, batch, nbatch) = gemm_io(c, bufs);
        let cfg = (v.0[0], v.0[1], v.0[2], v.0[3], v.0[4]);
        b.encode_gemm_sg(cb, cfg, a, w, o, m, k, n, ap, bp, cp, batch, nbatch);
    }
}

pub struct GemmMlx;
impl Kernel for GemmMlx {
    fn name(&self) -> &'static str {
        "gemm_mlx"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Contraction
    }
    fn admits(&self, c: &Composition) -> bool {
        gemm_mlx_ok(c)
    }
    fn variants(&self, _c: &Composition) -> Vec<Variant> {
        vec![Variant([255, 0, 0, 0, 0])]
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], _v: Variant, c: &Composition) {
        let (a, w, o, m, k, n, ap, bp, cp, batch, nbatch) = gemm_io(c, bufs);
        b.encode_gemm_mlx(cb, a, w, o, m, k, n, ap, bp, cp, batch, nbatch);
    }
}

fn attn_dispatch(b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], v: Variant, c: &Composition) {
    let a = c.flash();
    b.encode_attn(cb, v.0[0], v.0[1], v.0[2], &bufs[0], &bufs[1], &bufs[2], &bufs[3], &bufs[4], &a.p);
}

pub struct AttnOnline;
impl Kernel for AttnOnline {
    fn name(&self) -> &'static str {
        "attn_online"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Flash
    }
    fn admits(&self, _c: &Composition) -> bool {
        true
    }
    fn variants(&self, c: &Composition) -> Vec<Variant> {
        let a = c.flash();
        let ctx = a.p[3] as u32;
        let vec = if a.vec_ok { 4 } else { 1 };
        [1u32, 2, 4, 8]
            .iter()
            .filter(|s| **s <= ctx)
            .map(|s| Variant([0, *s, vec, 0, 0]))
            .collect()
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], v: Variant, c: &Composition) {
        attn_dispatch(b, cb, bufs, v, c);
    }
}

pub struct AttnFlash;
impl Kernel for AttnFlash {
    fn name(&self) -> &'static str {
        "attn_flash"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Flash
    }
    fn admits(&self, c: &Composition) -> bool {
        !crate::atoms::flash_candidates(&c.flash().p).is_empty()
    }
    fn variants(&self, c: &Composition) -> Vec<Variant> {
        crate::atoms::flash_candidates(&c.flash().p)
            .into_iter()
            .map(|(sgs, bk)| Variant([1, sgs, bk, 0, 0]))
            .collect()
    }
    fn dispatch(&self, b: &CuteMetal, cb: &CommandBufferRef, bufs: &[Buffer], v: Variant, c: &Composition) {
        attn_dispatch(b, cb, bufs, v, c);
    }
}

pub struct MlxMatmul;
impl Kernel for MlxMatmul {
    fn name(&self) -> &'static str {
        "mlx_matmul"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Contraction
    }
    fn admits(&self, _c: &Composition) -> bool {
        false
    }
    fn breaks_cb(&self) -> bool {
        true
    }
    fn variants(&self, _c: &Composition) -> Vec<Variant> {
        vec![Variant([255, 0, 0, 0, 0])]
    }
    fn dispatch(&self, _b: &CuteMetal, _cb: &CommandBufferRef, bufs: &[Buffer], _v: Variant, c: &Composition) {
        let s = c.contraction();
        unsafe {
            let a = (bufs[0].contents() as *const f32).add(s.ap[0] as usize);
            let w = (bufs[1].contents() as *const f32).add(s.bp[0] as usize);
            let o = (bufs[2].contents() as *mut f32).add(s.cp[0] as usize);
            crate::mlx_ffi::mlx_matmul(a, w, o, s.m, c.kext() as usize, s.n);
        }
    }
}

pub struct MlxSdpa;
impl Kernel for MlxSdpa {
    fn name(&self) -> &'static str {
        "mlx_sdpa"
    }
    fn servicer(&self) -> ServicerTag {
        ServicerTag::Flash
    }
    fn admits(&self, _c: &Composition) -> bool {
        false
    }
    fn breaks_cb(&self) -> bool {
        true
    }
    fn variants(&self, _c: &Composition) -> Vec<Variant> {
        vec![Variant([0, 0, 0, 0, 0])]
    }
    fn dispatch(&self, _b: &CuteMetal, _cb: &CommandBufferRef, bufs: &[Buffer], _v: Variant, c: &Composition) {
        let a = c.flash();
        crate::mlx_ffi::mlx_sdpa(
            bufs[0].contents() as *const f32,
            bufs[1].contents() as *const f32,
            bufs[2].contents() as *const f32,
            bufs[3].contents() as *const f32,
            bufs[4].contents() as *mut f32,
            &a.p,
        );
    }
}

pub fn registry() -> Vec<Box<dyn Kernel>> {
    vec![
        Box::new(GemvT),
        Box::new(GemvSg),
        Box::new(GemvMlx),
        Box::new(GemvRows),
        Box::new(ScalarMatmul),
        Box::new(BmmScalar),
        Box::new(GemmSg),
        Box::new(GemmMlx),
        Box::new(AttnOnline),
        Box::new(AttnFlash),
        Box::new(MlxMatmul),
        Box::new(MlxSdpa),
    ]
}
