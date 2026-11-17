use cute_layout::algebra::tiling::zipped_divide;
use cute_layout::Layout;
use egglog::sort::S;
use egglog::{CommandOutput, EGraph, Term};

pub struct Binding {
    pub spec: String,
    pub source: String,
    pub access: String,
}

fn tile_at_path(layout: &Layout, path: &[usize], tile: &[i64]) -> Option<Layout> {
    match path.split_first() {
        None => zipped_divide(layout, tile).ok(),
        Some((&i, rest)) => {
            let mut modes = layout.modes();
            if i >= modes.len() {
                return None;
            }
            modes[i] = tile_at_path(&modes[i], rest, tile)?;
            Some(Layout::from_modes(modes))
        }
    }
}

fn mode_tile(data: &str, spec: &str) -> Option<String> {
    let (ps, ts) = spec.split_once('|')?;
    let path: Vec<usize> = if ps.trim().is_empty() {
        Vec::new()
    } else {
        ps.split('.')
            .map(|x| x.trim().parse().ok())
            .collect::<Option<_>>()?
    };
    let tile: Vec<i64> = ts
        .split(',')
        .map(|x| x.trim().parse().ok())
        .collect::<Option<_>>()?;
    let layout: Layout = data.parse().ok()?;
    Some(tile_at_path(&layout, &path, &tile)?.to_string())
}

fn register(eg: &mut EGraph) {
    egglog::add_primitive!(eg, "cute-tile-mode" = |data: S, spec: S| -?> S {{
        mode_tile(data.as_str(), spec.as_str()).map(|s| s.into())
    }});
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub name: &'static str,
    pub rank: usize,
    pub tiles: Vec<Vec<i64>>,
}

#[derive(Debug, Clone)]
pub struct Choice {
    pub frame: &'static str,
    pub tile: Vec<i64>,
    pub access: String,
    pub waste: f64,
    pub levels: usize,
}

fn leaf_tiles(access: &str) -> Option<Vec<i64>> {
    let l: Layout = access.parse().ok()?;
    Some(
        l.modes()
            .iter()
            .map(|m| m.modes().first().map(|s| s.size()).unwrap_or_else(|| m.size()))
            .collect(),
    )
}

fn embeds(leaves: &[i64], tile: &[i64]) -> bool {
    let mut k = 0usize;
    for l in leaves {
        if k < tile.len() && *l == tile[k] {
            k += 1;
        }
    }
    k == tile.len()
}

fn levels(access: &str) -> usize {
    access
        .split(':')
        .next()
        .unwrap_or("")
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .count()
}

pub fn candidates(data: &str, bindings: &[Binding], frames: &[Frame]) -> Vec<Choice> {
    let Some(orig) = data.parse::<Layout>().ok().map(|l| l.size() as f64) else {
        return Vec::new();
    };
    let mut seed = vec![Binding {
        spec: String::new(),
        source: String::new(),
        access: data.to_string(),
    }];
    seed.extend(bindings.iter().map(|b| Binding {
        spec: b.spec.clone(),
        source: b.source.clone(),
        access: b.access.clone(),
    }));
    let mut out: Vec<Choice> = Vec::new();
    for b in &seed {
        let Some(leaves) = leaf_tiles(&b.access) else {
            continue;
        };
        let Ok(l) = b.access.parse::<Layout>() else {
            continue;
        };
        let waste = l.size() as f64 / orig;
        let lv = levels(&b.access);
        for f in frames {
            for tile in &f.tiles {
                if tile.len() != f.rank || !embeds(&leaves, tile) {
                    continue;
                }
                if out
                    .iter()
                    .any(|c| c.frame == f.name && c.tile == *tile && c.access == b.access)
                {
                    continue;
                }
                out.push(Choice {
                    frame: f.name,
                    tile: tile.clone(),
                    access: b.access.clone(),
                    waste,
                    levels: lv,
                });
            }
        }
    }
    out.sort_by(|a, b| {
        let (pa, pb): (i64, i64) = (a.tile.iter().product(), b.tile.iter().product());
        a.waste
            .partial_cmp(&b.waste)
            .unwrap()
            .then(pb.cmp(&pa))
            .then(a.levels.cmp(&b.levels))
    });
    out
}

pub fn extract(data: &str, bindings: &[Binding], frames: &[Frame]) -> Option<Choice> {
    candidates(data, bindings, frames).into_iter().next()
}

pub fn saturate(data: &str, specs: &[&str], steps: usize) -> Vec<Binding> {
    let mut eg = EGraph::default();
    register(&mut eg);

    let mut prog = String::new();
    prog.push_str("(relation Data (String))\n");
    prog.push_str("(relation Spec (String))\n");
    prog.push_str("(relation Serviced (String String String))\n");
    prog.push_str(
        "(rule ((Data ?l) (Spec ?s) (= ?a (cute-tile-mode ?l ?s))) ((Serviced ?s ?l ?a) (Data ?a)))\n",
    );
    prog.push_str(&format!("(Data \"{data}\")\n"));
    for s in specs {
        prog.push_str(&format!("(Spec \"{s}\")\n"));
    }
    prog.push_str(&format!("(run {steps})\n"));
    eg.parse_and_run_program(None, &prog)
        .expect("egglog saturation");

    let out = eg
        .parse_and_run_program(None, "(print-function Serviced 1000000)")
        .expect("print serviced");

    let mut bindings = Vec::new();
    for o in out {
        if let CommandOutput::PrintFunction(_f, dag, rows, _mode) = o {
            for (key, _val) in rows {
                if let Term::App(name, children) = dag.get(key) {
                    if name.as_str() == "Serviced" && children.len() == 3 {
                        let kids = children.clone();
                        let field =
                            |i: usize| dag.to_string(kids[i]).trim_matches('"').to_string();
                        bindings.push(Binding {
                            spec: field(0),
                            source: field(1),
                            access: field(2),
                        });
                    }
                }
            }
        }
    }
    bindings
}
