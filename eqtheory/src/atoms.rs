pub struct MmaAtom {
    pub name: &'static str,
    pub tm: i64,
    pub tn: i64,
    pub tk: i64,
}

pub struct MmaPlan {
    pub a_trans: bool,
    pub b_trans: bool,
}

pub const SG_F32_8X8X8: MmaAtom = MmaAtom {
    name: "simdgroup_float8x8",
    tm: 8,
    tn: 8,
    tk: 8,
};

pub const MMA_ATOMS: &[&MmaAtom] = &[&SG_F32_8X8X8];

pub fn mma_plan(
    atom: &MmaAtom,
    m: usize,
    k: usize,
    n: usize,
    ap: [i32; 4],
    bp: [i32; 4],
    cp: [i32; 4],
) -> Option<MmaPlan> {
    if m < 2 {
        return None;
    }
    if m as i64 % atom.tm != 0 || n as i64 % atom.tn != 0 || k as i64 % atom.tk != 0 {
        return None;
    }
    if cp[2] != 1 || cp[1] < 1 {
        return None;
    }
    let a_trans = if ap[2] == 1 {
        false
    } else if ap[1] == 1 {
        true
    } else {
        return None;
    };
    let b_trans = if bp[2] == 1 {
        false
    } else if bp[1] == 1 {
        true
    } else {
        return None;
    };
    let lda = if a_trans { ap[2] } else { ap[1] };
    let ldb = if b_trans { bp[2] } else { bp[1] };
    if lda < 1 || ldb < 1 {
        return None;
    }
    Some(MmaPlan { a_trans, b_trans })
}

pub fn mma_candidates(m: usize, n: usize) -> Vec<(u32, u32, u32, u32)> {
    let mut v = Vec::new();
    for (sgm, sgn) in [(1u32, 1u32), (2, 1), (2, 2), (4, 2)] {
        for (atm, atn) in [(2u32, 2u32), (4, 2), (2, 4), (4, 4)] {
            let bm = (sgm * atm * 8) as usize;
            let bn = (sgn * atn * 8) as usize;
            if bm > 128 || bn > 128 {
                continue;
            }
            if m % bm != 0 || n % bn != 0 {
                continue;
            }
            v.push((sgm, sgn, atm, atn));
        }
    }
    v
}
