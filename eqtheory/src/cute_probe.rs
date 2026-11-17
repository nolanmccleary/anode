use luminal::logical_ir::MapEntry;

pub fn eval_entry(e: &MapEntry, coords: &[i64]) -> Option<i64> {
    match e {
        MapEntry::Coord { from_end, .. } => coords.get(coords.len().checked_sub(1 + from_end)?).copied(),
        MapEntry::Lit(x) => Some(x.to_usize()? as i64),
        MapEntry::Add(a, b) => Some(eval_entry(a, coords)? + eval_entry(b, coords)?),
        MapEntry::Mul(a, n) => Some(eval_entry(a, coords)? * n.to_usize()? as i64),
        MapEntry::Div(a, n) => Some(eval_entry(a, coords)? / n.to_usize()? as i64),
        MapEntry::Rem(a, n) => Some(eval_entry(a, coords)? % n.to_usize()? as i64),
        MapEntry::Min(a, b) => Some(eval_entry(a, coords)?.min(eval_entry(b, coords)?)),
        MapEntry::Max(a, b) => Some(eval_entry(a, coords)?.max(eval_entry(b, coords)?)),
    }
}

pub fn flat_index(entries: &[MapEntry], coords: &[i64], src_strides: &[i64]) -> Option<i64> {
    let mut acc = 0i64;
    for (e, s) in entries.iter().zip(src_strides) {
        acc += eval_entry(e, coords)? * s;
    }
    Some(acc)
}

pub fn factor(vals: &[i64]) -> Option<Vec<(i64, i64)>> {
    let n = vals.len() as i64;
    if n <= 1 {
        return Some(vec![]);
    }
    let d = vals[1] - vals[0];
    let mut s = n;
    'outer: while s >= 2 {
        if n % s == 0 {
            for a in 0..(n / s) {
                let base = vals[(a * s) as usize];
                for b in 0..s {
                    if vals[(a * s + b) as usize] != base + b * d {
                        s -= 1;
                        continue 'outer;
                    }
                }
            }
            break;
        }
        s -= 1;
    }
    if s < 2 {
        return None;
    }
    let mut modes = vec![(s, d)];
    if s < n {
        let sub: Vec<i64> = vals.iter().step_by(s as usize).copied().collect();
        modes.extend(factor(&sub)?);
    }
    Some(modes)
}

pub fn mode_str(modes: &[(i64, i64)]) -> (String, String) {
    if modes.is_empty() {
        return ("1".into(), "0".into());
    }
    if modes.len() == 1 {
        return (modes[0].0.to_string(), modes[0].1.to_string());
    }
    let sh: Vec<String> = modes.iter().map(|m| m.0.to_string()).collect();
    let st: Vec<String> = modes.iter().map(|m| m.1.to_string()).collect();
    (format!("({})", sh.join(",")), format!("({})", st.join(",")))
}

pub struct Probed {
    pub layout: String,
    pub offset: i64,
}

pub fn probe(entries: &[MapEntry], out_dims: &[i64], src_strides: &[i64]) -> Option<Probed> {
    let rank = out_dims.len();
    let zero = vec![0i64; rank];
    let offset = flat_index(entries, &zero, src_strides)?;
    let mut axis_modes: Vec<Vec<(i64, i64)>> = Vec::new();
    for ax in 0..rank {
        let mut vals = Vec::with_capacity(out_dims[ax] as usize);
        let mut c = zero.clone();
        for v in 0..out_dims[ax] {
            c[ax] = v;
            vals.push(flat_index(entries, &c, src_strides)?);
        }
        axis_modes.push(factor(&vals)?);
    }
    let total: i64 = out_dims.iter().product();
    let predict = |c: &[i64]| -> i64 {
        let mut acc = offset;
        for (ax, modes) in axis_modes.iter().enumerate() {
            let mut v = c[ax];
            for (s, d) in modes {
                acc += (v % s) * d;
                v /= s;
            }
        }
        acc
    };
    let check = |c: &[i64]| flat_index(entries, c, src_strides) == Some(predict(c));
    if total <= 1 << 16 {
        let mut c = vec![0i64; rank];
        for i in 0..total {
            let mut r = i;
            for ax in 0..rank {
                c[ax] = r % out_dims[ax];
                r /= out_dims[ax];
            }
            if !check(&c) {
                return None;
            }
        }
    } else {
        let mut state = 0x9e3779b97f4a7c15u64;
        for _ in 0..4096 {
            let c: Vec<i64> = out_dims
                .iter()
                .map(|&d| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) as i64).rem_euclid(d)
                })
                .collect();
            if !check(&c) {
                return None;
            }
        }
    }
    let parts: Vec<(String, String)> = axis_modes.iter().map(|m| mode_str(m)).collect();
    let sh: Vec<&str> = parts.iter().map(|p| p.0.as_str()).collect();
    let st: Vec<&str> = parts.iter().map(|p| p.1.as_str()).collect();
    Some(Probed {
        layout: format!("({}):({})", sh.join(","), st.join(",")),
        offset,
    })
}

