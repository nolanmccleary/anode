use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex};
use luminal::shape::IntExpr;
use luminal_metal::HostBuffer;
use model_zoo::mini::llama3::MiniLlama3;

#[derive(Clone, Copy)]
pub struct ModelConfig {
    pub vocab: usize,
    pub d: usize,
    pub ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub cache_capacity: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            vocab: 32_000,
            d: 512,
            ff: 1376,
            n_heads: 8,
            n_kv_heads: 2,
            layers: 2,
            cache_capacity: 4,
        }
    }
}

pub struct Loaded {
    pub graph: Graph,
    pub outputs: Vec<GraphTensor>,
    fixed: Vec<(NodeIndex, Vec<i32>)>,
}

pub fn load(cfg: &ModelConfig) -> Loaded {
    let mut cx = Graph::new();
    let model = MiniLlama3::new(
        cfg.vocab,
        cfg.d,
        cfg.ff,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.layers,
        &mut cx,
    );
    let head_dim = cfg.d / cfg.n_heads;
    let kv_dim = cfg.n_kv_heads * head_dim;

    let ids = cx.tensor(vec![1], DType::Int);
    let caches: Vec<(GraphTensor, GraphTensor)> = (0..cfg.layers)
        .map(|_| {
            (
                cx.tensor(vec![cfg.cache_capacity, kv_dim], DType::F32),
                cx.tensor(vec![cfg.cache_capacity, kv_dim], DType::F32),
            )
        })
        .collect();
    let gather = cx.tensor(vec![2], DType::Int);
    let scatter = cx.tensor(vec![1], DType::Int);

    let (logits, cache_outputs) =
        model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize));

    let mut outputs = vec![logits];
    outputs.extend(cache_outputs.into_iter().flat_map(|(kk, vv)| [kk, vv]));

    Loaded {
        graph: cx,
        outputs,
        fixed: vec![
            (ids.id, vec![3]),
            (gather.id, vec![0, 1]),
            (scatter.id, vec![1]),
        ],
    }
}

pub fn load_default() -> Loaded {
    load(&ModelConfig::default())
}

pub fn load_deep_views(n: usize, depth: usize) -> Loaded {
    let dim = IntExpr::from;
    let ni = n as i64;
    let mut cx = Graph::new();
    let x = cx.tensor([dim(ni), dim(ni)], DType::F32);
    let mut t = x;
    for _ in 0..depth {
        t = t.transpose(0, 1).merge_dims(0, 1).split_dims(0, dim(ni));
    }
    let w = cx.tensor([dim(ni), dim(ni)], DType::F32);
    let out: GraphTensor = t.matmul(w);
    Loaded {
        graph: cx,
        outputs: vec![out],
        fixed: Vec::new(),
    }
}

fn test_f32(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

impl Loaded {
    pub fn read_ids(&self) -> Vec<NodeIndex> {
        self.outputs.iter().map(|o| o.id).collect()
    }

    pub fn inputs(&self) -> FxHashMap<NodeIndex, HostBuffer> {
        let fixed: FxHashMap<NodeIndex, &Vec<i32>> =
            self.fixed.iter().map(|(id, v)| (*id, v)).collect();
        self.graph
            .logical
            .input_specs()
            .into_iter()
            .enumerate()
            .map(|(seed, spec)| {
                let hb = if let Some(v) = fixed.get(&spec.id) {
                    HostBuffer::from((*v).clone())
                } else {
                    let n: usize = spec.dims.iter().map(|d| d.to_usize().unwrap()).product();
                    match spec.dtype {
                        DType::F32 => HostBuffer::from(test_f32(n, seed)),
                        DType::Int if n == 1 => HostBuffer::from(vec![1i32]),
                        DType::Int => HostBuffer::from((0..n as i32).collect::<Vec<_>>()),
                        DType::I64 if n == 1 => HostBuffer::from(vec![1i64]),
                        DType::I64 => HostBuffer::from((0..n as i64).collect::<Vec<_>>()),
                        other => panic!("unsupported loader input dtype {other:?}"),
                    }
                };
                (spec.id, hb)
            })
            .collect()
    }
}
