use cute_layout::algebra::mapping::coalesce;
use cute_layout::algebra::tiling::zipped_divide;
use cute_layout::{IntTuple, Layout};

pub struct Kernel {
    pub name: &'static str,
    pub rank: usize,
    pub tilings: Vec<(&'static str, Vec<i64>)>,
}

pub struct Fit {
    pub tiling: &'static str,
    pub access: Layout,
    pub waste: f64,
}

fn row_major(extents: &[i64]) -> Layout {
    Layout::row_major(IntTuple::Tuple(
        extents.iter().map(|e| IntTuple::Leaf(*e)).collect(),
    ))
}

impl Kernel {
    pub fn bind(&self, extents: &[i64]) -> Vec<Fit> {
        if extents.len() != self.rank {
            return Vec::new();
        }
        let real: i64 = extents.iter().product();
        self.tilings
            .iter()
            .filter(|(_, t)| t.len() == self.rank)
            .filter_map(|(name, t)| {
                let tiled = zipped_divide(&row_major(extents), t).ok()?;
                Some(Fit {
                    tiling: name,
                    waste: tiled.size() as f64 / real as f64,
                    access: coalesce(&tiled),
                })
            })
            .collect()
    }
}
