//! FlashAttention as a SHARED primitive — the fair-comparison build.
//!
//! The honest setup (per the fairness constraint): flash is a backend kernel that
//! BOTH a baseline-Luminal and our composed system would register and select via
//! the same egglog+GA search (write the kernel, `union` it into the attention
//! e-class, the GA keeps it only if it profiles faster). So "flash vs naive" is a
//! SHARED choice, not our win — and the search picks whichever is actually faster
//! for the shape. The ONLY thing our system adds over baseline-with-flash is
//! autotuning flash's internal tile config (BR,BC). We sweep sequence length to
//! show the search would pick naive at small S and flash at large S — no
//! preferential access to flash for either side.
//!
//! Non-causal, single head, Dh=64. Run:
//!   cargo +1.99.1 run --release --example flash_attention -p luminal_metal
#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions, MTLSize};

const DH: usize = 64;
const SCALE: f32 = 0.125; // 1/sqrt(64)

fn cpu_attention(s: usize, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut o = vec![0f32; s * DH];
    for i in 0..s {
        let mut sc = vec![0f32; s];
        let mut m = f32::NEG_INFINITY;
        for j in 0..s {
            let mut dot = 0f32;
            for d in 0..DH { dot += q[i * DH + d] * k[j * DH + d]; }
            sc[j] = dot * SCALE;
            m = m.max(sc[j]);
        }
        let mut sum = 0f32;
        for j in 0..s { sc[j] = (sc[j] - m).exp(); sum += sc[j]; }
        for d in 0..DH {
            let mut acc = 0f32;
            for j in 0..s { acc += (sc[j] / sum) * v[j * DH + d]; }
            o[i * DH + d] = acc;
        }
    }
    o
}

/// Simdgroup-tiled flash: BR/8 simdgroups per threadgroup, each OWNING 8 query
/// rows end-to-end (no cross-simdgroup reduction). QK^T and P@V use the hardware
/// `simdgroup_float8x8` units; the online softmax runs between them on the 8 lanes
/// that own the rows. Loops over key tiles of BC; no S*S materialization.
/// (BR, BC) is the tile config our system autotunes.
fn flash_source(s: usize, br: usize, bc: usize) -> String {
    let nt = (br / 8) * 32; // threads = one simdgroup per 8 query rows
    format!(r#"#include <metal_stdlib>
using namespace metal;
kernel void flash(device const float* Q, device const float* K, device const float* V,
                  device float* O, uint tid [[thread_index_in_threadgroup]], uint bid [[threadgroup_position_in_grid]]) {{
    const uint DH={DH}u, S={s}u, BR={br}u, BC={bc}u, NT={nt}u; const float scale={SCALE}f;
    threadgroup float Qs[{br}*{DH}]; threadgroup float Ks[{bc}*{DH}]; threadgroup float Vs[{bc}*{DH}];
    threadgroup float Ss[{br}*{bc}]; threadgroup float Os[{br}*{DH}];
    threadgroup float ms[{br}]; threadgroup float ls[{br}];
    uint sg = tid / 32u, lane = tid % 32u, row0 = bid * BR;
    for (uint t=tid; t<BR*DH; t+=NT) {{ Qs[t]=Q[row0*DH+t]; Os[t]=0.0f; }}
    for (uint i=tid; i<BR; i+=NT) {{ ms[i]=-INFINITY; ls[i]=0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k0=0u; k0<S; k0+=BC) {{
        for (uint t=tid; t<BC*DH; t+=NT) {{ Ks[t]=K[k0*DH+t]; Vs[t]=V[k0*DH+t]; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Ss = Q @ K^T  — simdgroup sg owns rows [sg*8, sg*8+8)
        for (uint bc8=0u; bc8<BC/8u; ++bc8) {{
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.0f);
            for (uint dh8=0u; dh8<DH/8u; ++dh8) {{
                simdgroup_float8x8 qf, kf;
                simdgroup_load(qf, Qs + sg*8u*DH + dh8*8u, DH);
                simdgroup_load(kf, Ks + bc8*8u*DH + dh8*8u, DH, ulong2(0,0), true);
                simdgroup_multiply_accumulate(acc, qf, kf, acc);
            }}
            simdgroup_store(acc, Ss + sg*8u*BC + bc8*8u, BC);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax: lane<8 of simdgroup sg owns row sg*8+lane
        if (lane < 8u) {{
            uint i = sg*8u + lane; float m_old=ms[i], mx=m_old;
            for (uint j=0u;j<BC;++j) mx = max(mx, Ss[i*BC+j]*scale);
            float corr=exp(m_old-mx), rs=0.0f;
            for (uint j=0u;j<BC;++j) {{ float p=exp(Ss[i*BC+j]*scale-mx); Ss[i*BC+j]=p; rs+=p; }}
            ls[i]=corr*ls[i]+rs; ms[i]=mx;
            for (uint d=0u; d<DH; ++d) Os[i*DH+d] *= corr;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Os += P @ V  — simdgroup sg owns rows [sg*8, sg*8+8)
        for (uint dh8=0u; dh8<DH/8u; ++dh8) {{
            simdgroup_float8x8 acc; simdgroup_load(acc, Os + sg*8u*DH + dh8*8u, DH);
            for (uint bc8=0u; bc8<BC/8u; ++bc8) {{
                simdgroup_float8x8 pf, vf;
                simdgroup_load(pf, Ss + sg*8u*BC + bc8*8u, BC);
                simdgroup_load(vf, Vs + bc8*8u*DH + dh8*8u, DH);
                simdgroup_multiply_accumulate(acc, pf, vf, acc);
            }}
            simdgroup_store(acc, Os + sg*8u*DH + dh8*8u, DH);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (uint i=tid; i<BR; i+=NT) for (uint d=0u; d<DH; ++d) O[(row0+i)*DH+d] = Os[i*DH+d]/ls[i];
}}"#)
}

fn flash_valid(s: usize, br: usize, bc: usize) -> bool {
    // tiles are multiples of 8 (simdgroup); one simdgroup per 8 rows; smem < 32KB.
    br % 8 == 0 && bc % 8 == 0 && s % br == 0 && s % bc == 0
        && (br / 8) * 32 <= 1024
        && (2 * br * DH + 2 * bc * DH + br * bc + 2 * br) * 4 <= 32768
}

fn naive_sources(s: usize) -> (String, String, String) {
    (format!(r#"#include <metal_stdlib>
using namespace metal;
kernel void qk(device const float* Q, device const float* K, device float* Sc, uint gid [[thread_position_in_grid]]) {{
    const uint S={s}u, DH={DH}u; if (gid>=S*S) return; uint i=gid/S, j=gid%S; float d=0.0f;
    for (uint t=0u;t<DH;++t) d += Q[i*DH+t]*K[j*DH+t]; Sc[gid]=d*{SCALE}f; }}"#),
     format!(r#"#include <metal_stdlib>
using namespace metal;
kernel void sm(device float* Sc, uint i [[thread_position_in_grid]]) {{
    const uint S={s}u; if (i>=S) return; float m=-INFINITY;
    for (uint j=0u;j<S;++j) m=max(m,Sc[i*S+j]); float sum=0.0f;
    for (uint j=0u;j<S;++j){{ float e=exp(Sc[i*S+j]-m); Sc[i*S+j]=e; sum+=e; }}
    for (uint j=0u;j<S;++j) Sc[i*S+j]/=sum; }}"#),
     format!(r#"#include <metal_stdlib>
using namespace metal;
kernel void pv(device const float* Sc, device const float* V, device float* O, uint gid [[thread_position_in_grid]]) {{
    const uint S={s}u, DH={DH}u; if (gid>=S*DH) return; uint i=gid/DH, d=gid%DH; float acc=0.0f;
    for (uint j=0u;j<S;++j) acc += Sc[i*S+j]*V[j*DH+d]; O[gid]=acc; }}"#))
}

fn max_err(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) }

fn time_flash(dev: &Device, q: &metal::CommandQueue, s: usize, br: usize, bc: usize,
              qb: &metal::Buffer, kb: &metal::Buffer, vb: &metal::Buffer, ob: &metal::Buffer) -> Option<(f64, Vec<f32>)> {
    if !flash_valid(s, br, bc) { return None; }
    let lib = dev.new_library_with_source(&flash_source(s, br, bc), &metal::CompileOptions::new()).ok()?;
    let func = lib.get_function("flash", None).ok()?;
    let pso = dev.new_compute_pipeline_state_with_function(&func).ok()?;
    let mut best = f64::MAX;
    for run in 0..20 {
        let t = std::time::Instant::now();
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(qb), 0); enc.set_buffer(1, Some(kb), 0);
        enc.set_buffer(2, Some(vb), 0); enc.set_buffer(3, Some(ob), 0);
        // BR/8 simdgroups (BR/8*32 threads) per query tile of BR rows.
        enc.dispatch_thread_groups(MTLSize::new((s / br) as u64, 1, 1), MTLSize::new(((br / 8) * 32) as u64, 1, 1));
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        if run >= 5 { best = best.min(t.elapsed().as_secs_f64()); }
    }
    let out = unsafe { std::slice::from_raw_parts(ob.contents() as *const f32, s * DH) }.to_vec();
    Some((best, out))
}

fn run(dev: &Device, qd: &metal::CommandQueue, s: usize, verify: bool) {
    let mk = |n: usize, seed: f32| -> Vec<f32> { (0..n).map(|i| (((i as f32 * seed) % 7.0) - 3.0) * 0.1).collect() };
    let (q, k, v) = (mk(s * DH, 1.3), mk(s * DH, 2.1), mk(s * DH, 0.7));
    let buf = |d: &[f32]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let (qb, kb, vb) = (buf(&q), buf(&k), buf(&v));
    let ob = dev.new_buffer((s * DH * 4) as u64, MTLResourceOptions::StorageModeShared);

    // naive materialized (3 kernels, round-trips S*S scores).
    let (qks, sms, pvs) = naive_sources(s);
    let pso = |src: &str, name: &str| {
        let lib = dev.new_library_with_source(src, &metal::CompileOptions::new()).unwrap();
        dev.new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap()).unwrap()
    };
    let (pqk, psm, ppv) = (pso(&qks, "qk"), pso(&sms, "sm"), pso(&pvs, "pv"));
    let scb = dev.new_buffer((s * s * 4) as u64, MTLResourceOptions::StorageModeShared);
    let ob2 = dev.new_buffer((s * DH * 4) as u64, MTLResourceOptions::StorageModeShared);
    let mut naive_t = f64::MAX;
    for run in 0..20 {
        let t = std::time::Instant::now();
        let cb = qd.new_command_buffer();
        let d1 = |p: &metal::ComputePipelineState, n: u64, bufs: &[&metal::Buffer]| {
            let e = cb.new_compute_command_encoder(); e.set_compute_pipeline_state(p);
            for (i, b) in bufs.iter().enumerate() { e.set_buffer(i as u64, Some(b), 0); }
            e.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(256, 1, 1)); e.end_encoding();
        };
        d1(&pqk, (s * s) as u64, &[&qb, &kb, &scb]);
        d1(&psm, s as u64, &[&scb]);
        d1(&ppv, (s * DH) as u64, &[&scb, &vb, &ob2]);
        cb.commit(); cb.wait_until_completed();
        if run >= 5 { naive_t = naive_t.min(t.elapsed().as_secs_f64()); }
    }

    // flash: fixed config + autotuned (BR,BC). Simdgroup tiles are multiples of 8.
    let fixed = (16usize, 16usize);
    let (ft, fout) = time_flash(dev, qd, s, fixed.0, fixed.1, &qb, &kb, &vb, &ob).expect("flash");
    let mut best: Option<(f64, (usize, usize))> = None;
    for br in [8, 16, 32, 64] { for bc in [8, 16, 32, 64] {
        if let Some((t, out)) = time_flash(dev, qd, s, br, bc, &qb, &kb, &vb, &ob) {
            let ok = !verify || max_err(&out, &cpu_attention(s, &q, &k, &v)) < 1e-3;
            if ok && best.is_none_or(|(bt, _)| t < bt) { best = Some((t, (br, bc))); }
        }
    }}
    let (bt, bc) = best.unwrap();
    let errnote = if verify {
        let want = cpu_attention(s, &q, &k, &v);
        format!("err {:.0e}", max_err(&fout, &want))
    } else { "—".into() };
    let picked = if naive_t < bt { "NAIVE" } else { "FLASH" };
    println!("  S={s:<5} | naive {:7.0}us | flash(fixed) {:7.0}us | flash(auto {:?}) {:7.0}us | search picks: {picked:<5} | flash corr {errnote}",
        naive_t * 1e6, ft * 1e6, bc, bt * 1e6);
}

fn main() {
    let dev = Device::system_default().expect("no Metal device");
    let qd = dev.new_command_queue();
    println!("device: {}   FlashAttention vs naive, Dh={DH}, single head\n", dev.name());
    println!("The SAME search would pick whichever profiles faster — no preferential flash access:");
    run(&dev, &qd, 512, true);
    run(&dev, &qd, 1024, true);
    run(&dev, &qd, 2048, false);
    run(&dev, &qd, 4096, false);
    println!("\nThe SAME fitness-driven search picks whichever is faster per shape — no");
    println!("preferential flash access for either system. Our fair edge is only autotuning");
    println!("flash's (BR,BC); flash itself is shared. Flash (simdgroup-tiled, multi-simdgroup)");
    println!("is human-engineered; the layout machinery tiles/autotunes it, it doesn't invent it.");
}
