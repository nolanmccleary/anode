use crate::cute_probe::factor;
use crate::metal_backend::CuteMetal;
use luminal::bufferize::{BufferId, BufferNode};
use luminal::dtype::DType;
use luminal::prelude::petgraph::algo::toposort;
use luminal::prelude::FxHashMap;
use luminal_metal::layouts::DecodedLayout;
use luminal_metal::MetalPlan;
use metal::Buffer;
use std::collections::HashMap;

pub struct Probe {
    pub axis_modes: Vec<Vec<(i64, i64)>>,
    pub offset: i64,
    pub extents: Vec<i64>,
}

pub fn probe_decoded(layout: &DecodedLayout) -> Option<Probe> {
    let extents: Vec<i64> = layout.literal_extents()?.iter().map(|x| *x as i64).collect();
    let rank = extents.len();
    let zero = vec![0usize; rank];
    let offset = layout.element_index(&zero).ok()? as i64;
    let mut axis_modes = Vec::with_capacity(rank);
    for ax in 0..rank {
        let mut vals = Vec::with_capacity(extents[ax] as usize);
        let mut c = zero.clone();
        for v in 0..extents[ax] {
            c[ax] = v as usize;
            vals.push(layout.element_index(&c).ok()? as i64);
        }
        axis_modes.push(factor(&vals)?);
    }
    let total: i64 = extents.iter().product();
    let predict = |c: &[usize]| -> i64 {
        let mut acc = offset;
        for (ax, modes) in axis_modes.iter().enumerate() {
            let mut v = c[ax] as i64;
            for (s, d) in modes {
                acc += (v % s) * d;
                v /= s;
            }
        }
        acc
    };
    let mut state = 0x9e3779b97f4a7c15u64;
    let samples = if total <= 4096 { total } else { 4096 };
    for t in 0..samples {
        let c: Vec<usize> = if total <= 4096 {
            let mut r = t;
            extents
                .iter()
                .map(|d| {
                    let x = (r % d) as usize;
                    r /= d;
                    x
                })
                .collect()
        } else {
            extents
                .iter()
                .map(|d| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) as i64).rem_euclid(*d) as usize
                })
                .collect()
        };
        if layout.element_index(&c).ok()? as i64 != predict(&c) {
            return None;
        }
    }
    Some(Probe { axis_modes, offset, extents })
}

pub fn probe_exprs(
    entries: &[luminal::index_expr::IotaExpr],
    out_extents: &[i64],
    src_strides: &[i64],
    src_off: i64,
) -> Option<Probe> {
    let rank = out_extents.len();
    let flat = |coords: &[usize]| -> i64 {
        let mut acc = src_off;
        for (e, st) in entries.iter().zip(src_strides) {
            acc += e.eval(coords) * st;
        }
        acc
    };
    let zero = vec![0usize; rank];
    let offset = flat(&zero);
    let mut axis_modes = Vec::with_capacity(rank);
    for ax in 0..rank {
        let mut vals = Vec::with_capacity(out_extents[ax] as usize);
        let mut c = zero.clone();
        for v in 0..out_extents[ax] {
            c[ax] = v as usize;
            vals.push(flat(&c));
        }
        axis_modes.push(factor(&vals)?);
    }
    let p = Probe { axis_modes, offset, extents: out_extents.to_vec() };
    let predict = |c: &[usize]| -> i64 {
        let mut acc = p.offset;
        for (ax, modes) in p.axis_modes.iter().enumerate() {
            let mut v = c[ax] as i64;
            for (s, d) in modes {
                acc += (v % s) * d;
                v /= s;
            }
        }
        acc
    };
    let total: i64 = out_extents.iter().product();
    let mut state = 0x243f6a8885a308d3u64;
    let samples = total.min(4096);
    for t in 0..samples {
        let c: Vec<usize> = if total <= 4096 {
            let mut r = t;
            out_extents
                .iter()
                .map(|d| {
                    let x = (r % d) as usize;
                    r /= d;
                    x
                })
                .collect()
        } else {
            out_extents
                .iter()
                .map(|d| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) as i64).rem_euclid(*d) as usize
                })
                .collect()
        };
        if flat(&c) != predict(&c) {
            return None;
        }
    }
    Some(p)
}

pub fn meta_from(p: &Probe) -> Option<[i32; 18]> {
    let mut modes: Vec<(i64, i64)> = Vec::new();
    for ax in (0..p.axis_modes.len()).rev() {
        modes.extend(p.axis_modes[ax].iter().copied());
    }
    modes.retain(|(s, _)| *s != 1);
    if modes.is_empty() {
        modes.push((1, 0));
    }
    if modes.len() > 8 {
        return None;
    }
    let mut meta = [0i32; 18];
    meta[0] = modes.len() as i32;
    meta[1] = i32::try_from(p.offset).ok()?;
    for (k, (s, d)) in modes.iter().enumerate() {
        meta[2 + k] = i32::try_from(*s).ok()?;
        meta[10 + k] = i32::try_from(*d).ok()?;
    }
    Some(meta)
}

fn contiguous(p: &Probe) -> bool {
    let mut expect = 1i64;
    for ax in (0..p.axis_modes.len()).rev() {
        for (s, d) in &p.axis_modes[ax] {
            if *s == 1 {
                continue;
            }
            if *d != expect {
                return false;
            }
            expect *= s;
        }
    }
    p.offset == 0
}

fn axis_strides(p: &Probe) -> Option<(Vec<i64>, i64)> {
    let mut out = Vec::with_capacity(p.extents.len());
    for (ax, modes) in p.axis_modes.iter().enumerate() {
        let ms: Vec<(i64, i64)> = modes.iter().filter(|(s, _)| *s != 1).copied().collect();
        match ms.len() {
            0 => out.push(0),
            1 if ms[0].0 == p.extents[ax] => out.push(ms[0].1),
            _ => return None,
        }
    }
    Some((out, p.offset))
}

fn group_stride(modes_fastest_first: &[(i64, i64)]) -> Option<i64> {
    let ms: Vec<(i64, i64)> = modes_fastest_first.iter().filter(|(s, _)| *s != 1).copied().collect();
    if ms.is_empty() {
        return Some(0);
    }
    if ms.iter().all(|(_, d)| *d == 0) {
        return Some(0);
    }
    let s0 = ms[0].1;
    let mut expect = s0;
    for (sz, d) in &ms {
        if *d != expect {
            return None;
        }
        expect = d * sz;
    }
    Some(s0)
}

enum Tape {
    Run1d {
        kernel: &'static str,
        bufs: Vec<Buffer>,
        metas: Vec<(u64, [i32; 18])>,
        scalars: Vec<(u64, u32)>,
        n: u64,
    },
    MetaOnly {
        kernel: &'static str,
        bufs: Vec<Buffer>,
        meta: [i32; 18],
        n: u64,
    },
    Matmul {
        a: Buffer,
        b: Buffer,
        c: Buffer,
        m: usize,
        k: usize,
        n: usize,
        ap: [i32; 4],
        bp: [i32; 4],
        cp: [i32; 4],
    },
}

pub struct AffineExec {
    pub backend: CuteMetal,
    bufs: HashMap<BufferId, Buffer>,
    tape: Vec<Tape>,
    outputs: Vec<(usize, BufferId, usize)>,
    pub dispatches: usize,
    pub gemms: usize,
    pub decomposed_reduces: usize,
    pub peak_bytes: usize,
    cur_bytes: usize,
    pub alloc_total: usize,
}

fn dl_bytes(l: &DecodedLayout) -> usize {
    let span = l.literal_span_elements().expect("literal span");
    let w = (l.width_bits() as usize).div_ceil(8);
    (span * w).max(4)
}

impl AffineExec {
    pub fn new() -> Self {
        AffineExec {
            backend: CuteMetal::new(),
            bufs: HashMap::new(),
            tape: Vec::new(),
            outputs: Vec::new(),
            dispatches: 0,
            gemms: 0,
            decomposed_reduces: 0,
            peak_bytes: 0,
            cur_bytes: 0,
            alloc_total: 0,
        }
    }

    fn alloc_counted(&mut self, bytes: usize) -> Buffer {
        self.alloc_total += bytes;
        self.backend.alloc_bytes(bytes)
    }

    fn buf(&self, id: &BufferId) -> &Buffer {
        self.bufs.get(id).expect("buffer allocated")
    }

    fn contiguous_or_copy(&mut self, src: Buffer, p: &Probe, bits: i64) -> Buffer {
        if contiguous(p) {
            return src;
        }
        let n: i64 = p.extents.iter().product();
        let bytes = (n as usize) * (bits as usize).div_ceil(8);
        let tmp = self.alloc_counted(bytes);
        let kernel = if bits == 32 {
            "copy_strided_i"
        } else {
            panic!("strided copy at width {bits} unsupported")
        };
        let meta = meta_from(p).expect("coord copy meta");
        self.push1d(
            kernel,
            vec![src, tmp.clone()],
            vec![(3, meta)],
            vec![(2, n as u32)],
            n as u64,
        );
        tmp
    }

    fn push1d(
        &mut self,
        kernel: &'static str,
        bufs: Vec<Buffer>,
        metas: Vec<(u64, [i32; 18])>,
        scalars: Vec<(u64, u32)>,
        n: u64,
    ) {
        self.tape.push(Tape::Run1d { kernel, bufs, metas, scalars, n });
        self.dispatches += 1;
    }

    pub fn build(
        &mut self,
        plan: &MetalPlan,
        specs: &[luminal::logical_ir::InputSpec],
        inputs: &FxHashMap<luminal::prelude::NodeIndex, luminal_metal::HostBuffer>,
    ) {
        let mut lit_of: HashMap<BufferId, i64> = HashMap::new();
        for (id, b) in plan.buffers.iter() {
            let bytes = dl_bytes(&b.layout);
            let nb = self.alloc_counted(bytes);
            self.bufs.insert(id.clone(), nb);
            if let Some(l) = b.lit {
                lit_of.insert(id.clone(), l);
            }
        }
        let order = toposort(&plan.dag, None).expect("plan is a DAG");
        for nid in order {
            match &plan.dag[nid] {
                BufferNode::BufferInput { slots } => {
                    assert_eq!(slots.len(), specs.len(), "input slot count matches specs");
                    for slot in slots.iter() {
                        let lit = *lit_of.get(&slot.buffer).expect("input boundary lit") as usize;
                        let spec = specs.get(lit).expect("lit indexes input specs");
                        let hb = inputs.get(&spec.id).expect("input host data");
                        let dst = self.buf(&slot.buffer);
                        let bytes = match spec.dtype {
                            DType::F32 => hb.as_f32().unwrap().len() * 4,
                            DType::Int => hb.as_i32().unwrap().len() * 4,
                            other => panic!("input dtype {other:?}"),
                        };
                        assert!(
                            dst.length() as usize >= bytes,
                            "input slot {} buffer {} bytes < host {} bytes — slot/spec order mismatch",
                            spec.label,
                            dst.length(),
                            bytes
                        );
                        unsafe {
                            match spec.dtype {
                                DType::F32 => {
                                    let v = hb.as_f32().unwrap();
                                    std::ptr::copy_nonoverlapping(
                                        v.as_ptr() as *const u8,
                                        dst.contents() as *mut u8,
                                        v.len() * 4,
                                    );
                                }
                                DType::Int => {
                                    let v = hb.as_i32().unwrap();
                                    std::ptr::copy_nonoverlapping(
                                        v.as_ptr() as *const u8,
                                        dst.contents() as *mut u8,
                                        v.len() * 4,
                                    );
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
                BufferNode::BufferOutput { slots } => {
                    for s in slots {
                        let span = s.layout.literal_span_elements().expect("output span");
                        self.outputs.push((s.index, s.buffer.clone(), span));
                    }
                }
                BufferNode::BufferCopy { .. } | BufferNode::BufferCopyValue { .. } => {
                    panic!("unexpected buffer copy node in this plan")
                }
                BufferNode::Compute { op, reads, writes, operand_info, result_info, .. } => {
                    if std::env::var("TRACE").is_ok() {
                        let l = op.label();
                        if l != "BufferAlloc" && l != "BufferFree" {
                            println!(
                                "tape[{}] <- {l} reads {:?} writes {:?}",
                                self.tape.len(),
                                reads.iter().map(|b| format!("{b:?}")).collect::<Vec<_>>(),
                                writes.iter().map(|b| format!("{b:?}")).collect::<Vec<_>>()
                            );
                        }
                    }
                    self.compute(op.as_ref(), reads, writes, operand_info, result_info);
                }
                other => panic!("unhandled plan node {other:?}"),
            }
        }
        self.outputs.sort_by_key(|(i, _, _)| *i);
    }

    fn compute(
        &mut self,
        op: &dyn luminal::buffer_tensor_ir::BufferTensorIrOp,
        reads: &[BufferId],
        writes: &[BufferId],
        operand_info: &[luminal::bufferize::SlotDescriptor<DecodedLayout>],
        result_info: &[luminal::bufferize::SlotDescriptor<DecodedLayout>],
    ) {
        let label = op.label().to_string();
        if std::env::var("PROBES").is_ok()
            && matches!(label.as_str(), "CopyGeneric" | "IndexMapApplyMaterialize" | "MulReduceSumGeneric")
            && self.tape.len() >= 15
            && self.tape.len() <= 18
        {
            for (k, sd) in operand_info.iter().enumerate() {
                if let Some(pp) = probe_decoded(&sd.layout) {
                    println!(
                        "tape[{}] {label} operand {k}: extents {:?} axis_modes {:?} offset {}",
                        self.tape.len(),
                        pp.extents,
                        pp.axis_modes,
                        pp.offset
                    );
                }
            }
            for (k, sd) in result_info.iter().enumerate() {
                if let Some(pp) = probe_decoded(&sd.layout) {
                    println!(
                        "tape[{}] {label} result {k}: extents {:?} axis_modes {:?}",
                        self.tape.len(),
                        pp.extents,
                        pp.axis_modes
                    );
                }
            }
        }
        let reads: &[BufferId] = if !writes.is_empty() && reads.last() == writes.first() {
            &reads[..reads.len() - 1]
        } else {
            reads
        };
        let probe_op = |k: usize| probe_decoded(&operand_info[k].layout).expect("operand probe");
        let out_span = |k: usize| result_info[k].layout.literal_span_elements().expect("span");
        match label.as_str() {
            "BufferAlloc" => {
                let bytes = dl_bytes(&result_info[0].layout);
                self.cur_bytes += bytes;
                self.peak_bytes = self.peak_bytes.max(self.cur_bytes);
            }
            "BufferFree" => {
                let bytes = dl_bytes(&operand_info[0].layout);
                self.cur_bytes = self.cur_bytes.saturating_sub(bytes);
            }
            "ConstantGeneric" => {
                let value = op
                    .as_any()
                    .downcast_ref::<luminal_metal::ops::constant::Constant>()
                    .map(|c| c.value)
                    .or_else(|| {
                        op.as_any()
                            .downcast_ref::<luminal_metal::ops::constant::ConstantDps>()
                            .map(|c| c.value)
                    })
                    .expect("Constant");
                let n = out_span(0);
                let host = vec![value as f32; n];
                let dst = self.buf(&writes[0]);
                assert!(dst.length() as usize >= n * 4, "constant buffer too small");
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host.as_ptr() as *const u8,
                        dst.contents() as *mut u8,
                        n * 4,
                    );
                }
            }
            "IotaGeneric" => {
                let expr_opt = op
                    .as_any()
                    .downcast_ref::<luminal_metal::ops::iota::Iota>()
                    .map(|i| i.expr.clone())
                    .or_else(|| {
                        op.as_any()
                            .downcast_ref::<luminal_metal::ops::iota::IotaDps>()
                            .map(|i| i.expr.clone())
                    })
                    .expect("Iota");
                let expr = expr_opt.expect("iota expr");
                let expr = &expr;
                let extents = result_info[0].layout.literal_extents().expect("iota extents");
                let rank = extents.len();
                let n: usize = extents.iter().product();
                let mut host = Vec::with_capacity(n);
                for i in 0..n {
                    let mut r = i;
                    let mut coords = vec![0usize; rank];
                    for ax in (0..rank).rev() {
                        coords[ax] = r % extents[ax];
                        r /= extents[ax];
                    }
                    host.push(expr.eval(&coords) as i32);
                }
                let dst = self.buf(&writes[0]);
                assert!(dst.length() as usize >= n * 4, "iota buffer too small");
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host.as_ptr() as *const u8,
                        dst.contents() as *mut u8,
                        n * 4,
                    );
                }
            }
            "AddFunctionalGeneric" | "MulFunctionalGeneric" => {
                let kernel = if label.starts_with("Add") { "add_strided" } else { "mul_strided" };
                let (pa, pb) = (probe_op(0), probe_op(1));
                let n = out_span(0) as u64;
                self.push1d(
                    kernel,
                    vec![self.buf(&reads[0]).clone(), self.buf(&reads[1]).clone(), self.buf(&writes[0]).clone()],
                    vec![(4, meta_from(&pa).expect("meta a")), (5, meta_from(&pb).expect("meta b"))],
                    vec![(3, n as u32)],
                    n,
                );
            }
            "SqrtFunctionalGeneric" | "RecipFunctionalGeneric" | "Exp2FunctionalGeneric"
            | "RsqrtFunctionalGeneric" => {
                let kernel = match label.as_str() {
                    "SqrtFunctionalGeneric" => "sqrt_s",
                    "RecipFunctionalGeneric" => "recip_s",
                    "Exp2FunctionalGeneric" => "exp2_s",
                    _ => "rsqrt_s",
                };
                let p = probe_op(0);
                let n = out_span(0) as u64;
                self.push1d(
                    kernel,
                    vec![self.buf(&reads[0]).clone(), self.buf(&writes[0]).clone()],
                    vec![(3, meta_from(&p).expect("meta"))],
                    vec![(2, n as u32)],
                    n,
                );
            }
            "CopyGeneric" | "IndexMapApplyMaterialize" => {
                let psrc = probe_op(0);
                let n = out_span(0) as u64;
                let read = if label == "IndexMapApplyMaterialize" {
                    let entries = op
                        .as_any()
                        .downcast_ref::<luminal_metal::ops::index_map_apply_materialize::IndexMapApplyMaterialize>()
                        .map(|o| o.entries.clone())
                        .or_else(|| {
                            op.as_any()
                                .downcast_ref::<luminal_metal::ops::index_map_apply_materialize::IndexMapApplyMaterializeDps>()
                                .map(|o| o.entries.clone())
                        })
                        .expect("materialize op")
                        .expect("index map beyond parsed subset");
                    let (sst, soff) = axis_strides(&psrc).expect("materialize source strides");
                    let out_ext: Vec<i64> = result_info[0]
                        .layout
                        .literal_extents()
                        .expect("materialize out extents")
                        .iter()
                        .map(|x| *x as i64)
                        .collect();
                    probe_exprs(&entries, &out_ext, &sst, soff).expect("materialize map probe")
                } else {
                    psrc
                };
                let kernel = match operand_info[0].layout.width_bits() {
                    32 => match operand_info[0].layout.dtype {
                        Some(luminal::dtype::PlanDtype::Int) => "copy_strided_i",
                        _ => "copy_strided",
                    },
                    other => panic!("copy width {other}"),
                };
                self.push1d(
                    kernel,
                    vec![self.buf(&reads[0]).clone(), self.buf(&writes[0]).clone()],
                    vec![(3, meta_from(&read).expect("meta"))],
                    vec![(2, n as u32)],
                    n,
                );
            }
            "CastGeneric" | "RoundingCastGeneric" => {
                let p = probe_op(0);
                let bits = operand_info[0].layout.width_bits();
                let kernel = match bits {
                    1 | 8 => "cast_b2f",
                    32 => "cast_i2f",
                    other => panic!("cast from width {other}"),
                };
                let sb = self.contiguous_or_copy(self.buf(&reads[0]).clone(), &p, bits);
                let n = out_span(0) as u64;
                self.push1d(
                    kernel,
                    vec![sb, self.buf(&writes[0]).clone()],
                    vec![],
                    vec![(2, n as u32)],
                    n,
                );
            }
            "LessThanGeneric" => {
                let (pa, pb) = (probe_op(0), probe_op(1));
                let ab = self.contiguous_or_copy(self.buf(&reads[0]).clone(), &pa, operand_info[0].layout.width_bits());
                let bb = self.contiguous_or_copy(self.buf(&reads[1]).clone(), &pb, operand_info[1].layout.width_bits());
                let n = out_span(0) as u64;
                self.push1d(
                    "less_than_i",
                    vec![ab, bb, self.buf(&writes[0]).clone()],
                    vec![],
                    vec![(3, n as u32)],
                    n,
                );
            }
            "ReduceSumGeneric" | "ReduceMaxGeneric" => {
                let any = op.as_any();
                let axis_from_end = if label.starts_with("ReduceSum") {
                    any.downcast_ref::<luminal_metal::ops::reduce_sum::ReduceSum>()
                        .map(|o| o.axis)
                        .or_else(|| {
                            any.downcast_ref::<luminal_metal::ops::reduce_sum::ReduceSumDps>()
                                .map(|o| o.axis)
                        })
                        .expect("ReduceSum") as usize
                } else {
                    any.downcast_ref::<luminal_metal::ops::reduce_max::ReduceMax>()
                        .map(|o| o.axis)
                        .or_else(|| {
                            any.downcast_ref::<luminal_metal::ops::reduce_max::ReduceMaxDps>()
                                .map(|o| o.axis)
                        })
                        .expect("ReduceMax") as usize
                };
                let kernel = if label.starts_with("ReduceSum") { "reduce_sum_axis" } else { "reduce_max_axis" };
                let p = probe_op(0);
                assert!(contiguous(&p), "reduce operand strided — extend mapping");
                let dims = &p.extents;
                let ax = dims.len() - 1 - axis_from_end;
                let outer: i64 = dims[..ax].iter().product();
                let axlen = dims[ax];
                let inner: i64 = dims[ax + 1..].iter().product();
                let n = (outer * inner) as u64;
                assert_eq!(n as usize, out_span(0), "reduce extents");
                self.push1d(
                    kernel,
                    vec![self.buf(&reads[0]).clone(), self.buf(&writes[0]).clone()],
                    vec![],
                    vec![(2, outer as u32), (3, axlen as u32), (4, inner as u32)],
                    n,
                );
            }
            "MulReduceSumGeneric" => {
                let axis_from_end = op
                    .as_any()
                    .downcast_ref::<luminal_metal::ops::mul_reduce_sum::MulReduceSum>()
                    .map(|o| o.axis)
                    .or_else(|| {
                        op.as_any()
                            .downcast_ref::<luminal_metal::ops::mul_reduce_sum::MulReduceSumDps>()
                            .map(|o| o.axis)
                    })
                    .expect("MulReduceSum") as usize;
                let (pa, pb) = (probe_op(0), probe_op(1));
                let dims = pa.extents.clone();
                let rank = dims.len();
                let ax = rank - 1 - axis_from_end;
                let varies = |p: &Probe, axis: usize| p.axis_modes[axis].iter().any(|(s, d)| *s != 1 && *d != 0);
                let mut gemm = pa.offset == 0 && pb.offset == 0 && varies(&pa, ax) && varies(&pb, ax);
                let (mut m_axes, mut n_axes) = (Vec::new(), Vec::new());
                if gemm {
                    for axis in 0..rank {
                        if axis == ax {
                            continue;
                        }
                        match (varies(&pa, axis), varies(&pb, axis)) {
                            (true, true) => {
                                gemm = false;
                                break;
                            }
                            (_, true) => n_axes.push(axis),
                            _ => m_axes.push(axis),
                        }
                    }
                }
                if gemm && m_axes.iter().any(|ma| n_axes.iter().any(|na| ma > na)) {
                    gemm = false;
                }
                let collect = |p: &Probe, axes: &[usize]| -> Vec<(i64, i64)> {
                    let mut v = Vec::new();
                    for axis in axes.iter().rev() {
                        v.extend(p.axis_modes[*axis].iter().copied());
                    }
                    v
                };
                let strides = if gemm {
                    match (
                        group_stride(&collect(&pa, &m_axes)),
                        group_stride(&collect(&pa, &[ax])),
                        group_stride(&collect(&pb, &[ax])),
                        group_stride(&collect(&pb, &n_axes)),
                    ) {
                        (Some(a), Some(b), Some(c), Some(d)) => Some((a, b, c, d)),
                        _ => None,
                    }
                } else {
                    None
                };
                if std::env::var("GEMMWHY").is_ok() && strides.is_none() {
                    println!(
                        "decomposed: dims {dims:?} ax {ax} a_modes {:?} b_modes {:?} gemm_flag {gemm} m{m_axes:?} n{n_axes:?}",
                        pa.axis_modes, pb.axis_modes
                    );
                }
                match strides {
                    Some((ars, acs, brs, bcs)) => {
                        let m: i64 = m_axes.iter().map(|x| dims[*x]).product();
                        let nn: i64 = n_axes.iter().map(|x| dims[*x]).product();
                        let k = dims[ax];
                        assert_eq!((m * nn) as usize, out_span(0), "gemm out span");
                        self.tape.push(Tape::Matmul {
                            a: self.buf(&reads[0]).clone(),
                            b: self.buf(&reads[1]).clone(),
                            c: self.buf(&writes[0]).clone(),
                            m: m as usize,
                            k: k as usize,
                            n: nn as usize,
                            ap: [0, ars as i32, acs as i32, 0],
                            bp: [0, brs as i32, bcs as i32, 0],
                            cp: [0, nn as i32, 1, 0],
                        });
                        self.dispatches += 1;
                        self.gemms += 1;
                    }
                    None => {
                        let total: i64 = dims.iter().product();
                        let tmp = self.alloc_counted(total as usize * 4);
                        self.push1d(
                            "mul_strided",
                            vec![self.buf(&reads[0]).clone(), self.buf(&reads[1]).clone(), tmp.clone()],
                            vec![(4, meta_from(&pa).expect("ma")), (5, meta_from(&pb).expect("mb"))],
                            vec![(3, total as u32)],
                            total as u64,
                        );
                        let outer: i64 = dims[..ax].iter().product();
                        let axlen = dims[ax];
                        let inner: i64 = dims[ax + 1..].iter().product();
                        let n = (outer * inner) as u64;
                        assert_eq!(n as usize, out_span(0), "decomposed reduce span");
                        self.push1d(
                            "reduce_sum_axis",
                            vec![tmp, self.buf(&writes[0]).clone()],
                            vec![],
                            vec![(2, outer as u32), (3, axlen as u32), (4, inner as u32)],
                            n,
                        );
                        self.decomposed_reduces += 1;
                    }
                }
            }
            "GatherGeneric" => {
                let pdata = probe_op(0);
                let (mut strides, doff) =
                    axis_strides(&pdata).expect("gather data not per-axis strided");
                let rank = op
                    .as_any()
                    .downcast_ref::<luminal_metal::ops::gather::Gather>()
                    .map(|g| g.rank)
                    .or_else(|| {
                        op.as_any()
                            .downcast_ref::<luminal_metal::ops::gather::GatherDps>()
                            .map(|g| g.rank)
                    })
                    .expect("Gather");
                while strides.len() < rank {
                    strides.insert(0, 0);
                }
                assert_eq!(strides.len(), rank, "gather data rank vs op rank");
                assert!(rank <= 4, "gather rank cap");
                assert_eq!(reads.len(), rank + 1, "gather operand count");
                let coords: Vec<Buffer> = (1..reads.len())
                    .map(|k| {
                        let pc = probe_op(k);
                        let bits = operand_info[k].layout.width_bits();
                        self.contiguous_or_copy(self.buf(&reads[k]).clone(), &pc, bits)
                    })
                    .collect();
                let n = out_span(0);
                let mut meta = [0i32; 18];
                meta[0] = rank as i32;
                meta[1] = n as i32;
                for (k, s) in strides.iter().enumerate() {
                    meta[2 + k] = *s as i32;
                }
                meta[6] = doff as i32;
                let c0 = coords[0].clone();
                let bufs = vec![
                    self.buf(&reads[0]).clone(),
                    c0.clone(),
                    coords.get(1).cloned().unwrap_or_else(|| c0.clone()),
                    coords.get(2).cloned().unwrap_or_else(|| c0.clone()),
                    coords.get(3).cloned().unwrap_or_else(|| c0.clone()),
                    self.buf(&writes[0]).clone(),
                ];
                self.tape.push(Tape::MetaOnly { kernel: "gather4", bufs, meta, n: n as u64 });
                self.dispatches += 1;
            }
            "ScatterFunctionalGeneric" => {
                let pinit = probe_op(0);
                let psrc = probe_op(1);
                assert!(contiguous(&psrc), "scatter src strided — extend mapping");
                let rank = pinit.extents.len();
                assert!(rank <= 4, "scatter rank cap");
                assert_eq!(reads.len(), rank + 2, "scatter operand count");
                let n = out_span(0) as u64;
                if reads[0] != writes[0] {
                    self.push1d(
                        "copy_strided",
                        vec![self.buf(&reads[0]).clone(), self.buf(&writes[0]).clone()],
                        vec![(3, meta_from(&pinit).expect("init meta"))],
                        vec![(2, n as u32)],
                        n,
                    );
                }
                let src_n: usize = psrc.extents.iter().product::<i64>() as usize;
                let pres = probe_decoded(&result_info[0].layout).expect("scatter result probe");
                let (strides, woff) =
                    axis_strides(&pres).expect("scatter result not per-axis strided");
                let mut meta = [0i32; 18];
                meta[0] = rank as i32;
                meta[1] = src_n as i32;
                for (k, s) in strides.iter().enumerate() {
                    meta[2 + k] = *s as i32;
                }
                meta[6] = woff as i32;
                let coords: Vec<Buffer> = (2..reads.len())
                    .map(|k| {
                        let pc = probe_op(k);
                        let bits = operand_info[k].layout.width_bits();
                        self.contiguous_or_copy(self.buf(&reads[k]).clone(), &pc, bits)
                    })
                    .collect();
                let c0 = coords[0].clone();
                let bufs = vec![
                    self.buf(&reads[1]).clone(),
                    c0.clone(),
                    coords.get(1).cloned().unwrap_or_else(|| c0.clone()),
                    coords.get(2).cloned().unwrap_or_else(|| c0.clone()),
                    coords.get(3).cloned().unwrap_or_else(|| c0.clone()),
                    self.buf(&writes[0]).clone(),
                ];
                self.tape.push(Tape::MetaOnly { kernel: "scatter4", bufs, meta, n: src_n as u64 });
                self.dispatches += 1;
            }
            other => panic!("plan op {other} has no shared-kernel mapping"),
        }
    }

    pub fn verify_tape(&self) {
        let mut writes: HashMap<u64, usize> = HashMap::new();
        for t in &self.tape {
            let out = match t {
                Tape::Run1d { bufs, .. } => bufs.last().unwrap(),
                Tape::MetaOnly { bufs, .. } => bufs.last().unwrap(),
                Tape::Matmul { c, .. } => c,
            };
            *writes.entry(out.gpu_address()).or_default() += 1;
        }
        let multi = writes.values().filter(|c| **c > 1).count();
        println!("buffers written more than once: {multi} (post-hoc verify unsound for those)");
        let readf = |b: &Buffer, n: usize| -> Vec<f32> { self.backend.read(b, n) };
        let off = |m: &[i32; 18], i: u64| -> i64 {
            let nm = m[0] as usize;
            let mut acc = m[1] as i64;
            let mut r = i as i64;
            for k in 0..nm {
                let sz = m[2 + k] as i64;
                acc += (r % sz) * m[10 + k] as i64;
                r /= sz;
            }
            acc
        };
        for (i, t) in self.tape.iter().enumerate() {
            let out_addr = match t {
                Tape::Run1d { bufs, .. } => bufs.last().unwrap().gpu_address(),
                Tape::MetaOnly { bufs, .. } => bufs.last().unwrap().gpu_address(),
                Tape::Matmul { c, .. } => c.gpu_address(),
            };
            if writes[&out_addr] > 1 {
                continue;
            }
            let bad: Option<String> = match t {
                Tape::Run1d { kernel, bufs, metas, scalars, n } => {
                    let n = *n as usize;
                    match *kernel {
                        "add_strided" | "mul_strided" => {
                            let (ma, mb) = (&metas[0].1, &metas[1].1);
                            let span = |m: &[i32; 18]| {
                                let nm = m[0] as usize;
                                let mut top = m[1] as i64;
                                for k in 0..nm {
                                    top += (m[2 + k] as i64 - 1) * (m[10 + k] as i64).max(0);
                                }
                                top as usize + 1
                            };
                            let a = readf(&bufs[0], span(ma));
                            let b = readf(&bufs[1], span(mb));
                            let c = readf(&bufs[2], n);
                            let mul = *kernel == "mul_strided";
                            (0..n).find_map(|j| {
                                let av = a[off(ma, j as u64) as usize];
                                let bv = b[off(mb, j as u64) as usize];
                                let e = if mul { av * bv } else { av + bv };
                                (e != c[j] && (e - c[j]).abs() > 1e-5 * e.abs().max(1.0)).then(|| {
                                    format!("j={j} expect {e} got {}", c[j])
                                })
                            })
                        }
                        "reduce_sum_axis" | "reduce_max_axis" => {
                            let (outer, axis, inner) = (
                                scalars[0].1 as usize,
                                scalars[1].1 as usize,
                                scalars[2].1 as usize,
                            );
                            let x = readf(&bufs[0], outer * axis * inner);
                            let o = readf(&bufs[1], outer * inner);
                            let is_sum = *kernel == "reduce_sum_axis";
                            (0..outer * inner).find_map(|j| {
                                let (ou, inn) = (j / inner, j % inner);
                                let mut acc = if is_sum { 0.0f32 } else { f32::NEG_INFINITY };
                                for a in 0..axis {
                                    let v = x[(ou * axis + a) * inner + inn];
                                    acc = if is_sum { acc + v } else { acc.max(v) };
                                }
                                (acc != o[j] && (acc - o[j]).abs() > 1e-4 * acc.abs().max(1.0))
                                    .then(|| format!("j={j} expect {acc} got {}", o[j]))
                            })
                        }
                        "copy_strided" | "copy_strided_i" => {
                            let m = &metas[0].1;
                            let span = {
                                let nm = m[0] as usize;
                                let mut top = m[1] as i64;
                                for k in 0..nm {
                                    top += (m[2 + k] as i64 - 1) * (m[10 + k] as i64).max(0);
                                }
                                top as usize + 1
                            };
                            if *kernel == "copy_strided" {
                                let a = readf(&bufs[0], span);
                                let c = readf(&bufs[1], n);
                                (0..n).find_map(|j| {
                                    let e = a[off(m, j as u64) as usize];
                                    (e != c[j]).then(|| format!("j={j} expect {e} got {}", c[j]))
                                })
                            } else {
                                let a = self.backend.read_i32(&bufs[0], span);
                                let c = self.backend.read_i32(&bufs[1], n);
                                (0..n).find_map(|j| {
                                    let e = a[off(m, j as u64) as usize];
                                    (e != c[j]).then(|| format!("j={j} expect {e} got {}", c[j]))
                                })
                            }
                        }
                        _ => None,
                    }
                }
                Tape::MetaOnly { kernel, bufs, meta, n } => {
                    let n = *n as usize;
                    let rank = meta[0] as usize;
                    let coords: Vec<Vec<i32>> = (0..rank)
                        .map(|k| self.backend.read_i32(&bufs[1 + k], n))
                        .collect();
                    let max_off = (0..n)
                        .map(|j| {
                            let mut o = meta[6] as i64;
                            for k in 0..rank {
                                o += coords[k][j] as i64 * meta[2 + k] as i64;
                            }
                            o
                        })
                        .max()
                        .unwrap_or(0) as usize;
                    if *kernel == "gather4" {
                        let data = readf(&bufs[0], max_off + 1);
                        let out = readf(&bufs[5], n);
                        (0..n).find_map(|j| {
                            let mut o = meta[6] as i64;
                            for k in 0..rank {
                                o += coords[k][j] as i64 * meta[2 + k] as i64;
                            }
                            let e = data[o as usize];
                            (e != out[j]).then(|| format!("j={j} off={o} expect {e} got {}", out[j]))
                        })
                    } else {
                        let src = readf(&bufs[0], n);
                        let out = readf(&bufs[5], max_off + 1);
                        (0..n).find_map(|j| {
                            let mut o = meta[6] as i64;
                            for k in 0..rank {
                                o += coords[k][j] as i64 * meta[2 + k] as i64;
                            }
                            let got = out[o as usize];
                            (src[j] != got)
                                .then(|| format!("j={j} off={o} expect {} got {got}", src[j]))
                        })
                    }
                }
                Tape::Matmul { a, b, c, m, k, n, ap, bp, cp } => {
                    let _ = cp;
                    let amax = (ap[1].max(0) as usize) * (m - 1)
                        + (ap[2].max(0) as usize) * (k - 1)
                        + 1;
                    let bmax = (bp[1].max(0) as usize) * (k - 1)
                        + (bp[2].max(0) as usize) * (n - 1)
                        + 1;
                    let av = readf(a, amax);
                    let bv = readf(b, bmax);
                    let cv = readf(c, m * n);
                    let mut found = None;
                    'outer: for mi in 0..*m {
                        for ni in 0..*n {
                            let mut acc = 0.0f32;
                            for ki in 0..*k {
                                acc += av[mi * ap[1] as usize + ki * ap[2] as usize]
                                    * bv[ki * bp[1] as usize + ni * bp[2] as usize];
                            }
                            let got = cv[mi * cp[1] as usize + ni * cp[2] as usize];
                            if (acc - got).abs() > 1e-3 * acc.abs().max(1.0) {
                                found = Some(format!("m={mi} n={ni} expect {acc} got {got}"));
                                break 'outer;
                            }
                        }
                    }
                    found
                }
                _ => None,
            };
            if let Some(msg) = bad {
                println!("FIRST DIVERGENCE at tape {i}: {msg}");
                self.dump_entry(i);
                return;
            }
        }
        println!("tape verified clean for checked kernel kinds");
    }

    fn dump_entry(&self, i: usize) {
        match &self.tape[i] {
            Tape::Run1d { kernel, metas, scalars, n, .. } => {
                println!("  {kernel} n={n} metas={metas:?} scalars={scalars:?}");
            }
            Tape::MetaOnly { kernel, meta, n, .. } => println!("  {kernel} n={n} meta={meta:?}"),
            Tape::Matmul { m, k, n, ap, bp, cp, .. } => {
                println!("  matmul m={m} k={k} n={n} ap={ap:?} bp={bp:?} cp={cp:?}")
            }
        }
    }

    pub fn hazards(&self) {
        for (i, t) in self.tape.iter().enumerate() {
            if let Tape::Run1d { kernel, bufs, metas, .. } = t {
                let out = bufs.last().unwrap();
                for (k, b) in bufs[..bufs.len() - 1].iter().enumerate() {
                    if b.gpu_address() == out.gpu_address() {
                        let strided = metas.iter().any(|(_, m)| {
                            let nm = m[0] as usize;
                            m[1] != 0
                                || (0..nm).any(|j| {
                                    let expect: i32 = m[2..2 + j].iter().product();
                                    m[10 + j] != expect && m[10 + j] != 0
                                })
                        });
                        println!(
                            "{i:>3} {kernel:<16} IN-PLACE operand {k}{}",
                            if strided { "  STRIDED READ — HAZARD" } else { "" }
                        );
                    }
                }
            }
        }
    }

    pub fn dump(&self) {
        for (i, t) in self.tape.iter().enumerate() {
            match t {
                Tape::Run1d { kernel, metas, n, .. } => {
                    let ms: Vec<String> = metas
                        .iter()
                        .map(|(_, m)| {
                            let nm = m[0] as usize;
                            format!(
                                "off{} sz{:?} st{:?}",
                                m[1],
                                &m[2..2 + nm],
                                &m[10..10 + nm]
                            )
                        })
                        .collect();
                    println!("{i:>3} {kernel:<16} n={n:<8} {}", ms.join(" | "));
                }
                Tape::MetaOnly { kernel, meta, n, .. } => {
                    println!("{i:>3} {kernel:<16} n={n:<8} rank={} strides={:?} off={}", meta[0], &meta[2..6], meta[6]);
                }
                Tape::Matmul { m, k, n, ap, bp, .. } => {
                    println!("{i:>3} tiled_matmul     m={m} k={k} n={n} ap={ap:?} bp={bp:?}");
                }
            }
        }
    }

    pub fn traffic_bytes(&self) -> usize {
        self.tape
            .iter()
            .map(|t| match t {
                Tape::Run1d { kernel, n, scalars, .. } => {
                    let n = *n as usize;
                    match *kernel {
                        "add_strided" | "mul_strided" => 3 * n * 4,
                        "copy_strided" | "copy_strided_i" | "cast_i2f" | "sqrt_s" | "recip_s"
                        | "exp2_s" | "rsqrt_s" => 2 * n * 4,
                        "cast_b2f" => n + n * 4,
                        "less_than_i" => 2 * n * 4 + n,
                        "reduce_sum_axis" | "reduce_max_axis" => {
                            let axis = scalars[1].1 as usize;
                            n * axis * 4 + n * 4
                        }
                        other => panic!("traffic for {other}"),
                    }
                }
                Tape::MetaOnly { n, .. } => {
                    let n = *n as usize;
                    n * 4 * 2 + n * 4 * 2
                }
                Tape::Matmul { m, k, n, .. } => (m * k + k * n + m * n) * 4,
            })
            .sum()
    }

    pub fn replay(&self) {
        let cb = self.backend.owned_command_buffer();
        for t in &self.tape {
            match t {
                Tape::Run1d { kernel, bufs, metas, scalars, n } => {
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(self.backend.pipeline(kernel));
                    for (k, b) in bufs.iter().enumerate() {
                        enc.set_buffer(k as u64, Some(b), 0);
                    }
                    for (slot, meta) in metas {
                        enc.set_bytes(*slot, 72, meta.as_ptr() as *const _);
                    }
                    for (slot, v) in scalars {
                        enc.set_bytes(*slot, 4, v as *const u32 as *const _);
                    }
                    let tg = 256u64.min((*n).max(1));
                    enc.dispatch_threads(
                        metal::MTLSize::new((*n).max(1), 1, 1),
                        metal::MTLSize::new(tg, 1, 1),
                    );
                    enc.end_encoding();
                }
                Tape::MetaOnly { kernel, bufs, meta, n } => {
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(self.backend.pipeline(kernel));
                    for (k, b) in bufs.iter().enumerate() {
                        enc.set_buffer(k as u64, Some(b), 0);
                    }
                    enc.set_bytes(6, 72, meta.as_ptr() as *const _);
                    let tg = 256u64.min((*n).max(1));
                    enc.dispatch_threads(
                        metal::MTLSize::new((*n).max(1), 1, 1),
                        metal::MTLSize::new(tg, 1, 1),
                    );
                    enc.end_encoding();
                }
                Tape::Matmul { a, b, c, m, k, n, ap, bp, cp } => {
                    self.backend.matmul_into(&cb, a, b, c, *m, *k, *n, *ap, *bp, *cp);
                }
            }
        }
        cb.commit();
        cb.wait_until_completed();
    }

    pub fn read_alloc(&self, id: u32, n: usize) -> Vec<f32> {
        let key = self
            .bufs
            .keys()
            .find(|k| format!("{k:?}") == format!("Allocated({id})"))
            .expect("alloc id");
        self.backend.read(&self.bufs[key], n)
    }

    pub fn dump_outputs(&self) {
        for (i, b, span) in &self.outputs {
            println!("output {i} <- {b:?} span {span}");
        }
    }

    pub fn read_output(&self, index: usize) -> Vec<f32> {
        let (_, buf, span) = self
            .outputs
            .iter()
            .find(|(i, _, _)| *i == index)
            .expect("output index");
        self.backend.read(self.buf(buf), *span)
    }
}
