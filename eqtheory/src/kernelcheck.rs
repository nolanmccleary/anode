use eqtheory::kernels::{EqClass, KERNELS};
use eqtheory::loaders::{self, ModelConfig};
use eqtheory::metal_backend::CuteMetal;
use std::collections::BTreeSet;

fn cfg() -> ModelConfig {
    ModelConfig {
        vocab: 32_000,
        d: 1024,
        ff: 2752,
        n_heads: 16,
        n_kv_heads: 4,
        layers: 2,
        cache_capacity: 4,
    }
}

fn required(constructor: &str) -> &'static [EqClass] {
    match constructor {
        "LogicalTensorInputLit" => &[],
        "LogicalConstant" => &[EqClass::Fill],
        "LogicalAdd" => &[EqClass::Add],
        "LogicalMul" => &[EqClass::Matmul, EqClass::MulEw],
        "LogicalReduceSum" => &[EqClass::Matmul, EqClass::ReduceSum],
        "LogicalReduceMax" => &[EqClass::ReduceMax],
        "LogicalRecip" => &[EqClass::Recip],
        "LogicalSqrt" => &[EqClass::Sqrt],
        "LogicalExp2" => &[EqClass::Exp2],
        "LogicalIota" => &[EqClass::Iota],
        "LogicalCast" => &[EqClass::CastI2F],
        "LogicalRoundingCast" => &[EqClass::Round],
        "LogicalGather" => &[EqClass::Gather],
        "LogicalScatter" => &[EqClass::Scatter],
        "LogicalLessThan" => &[EqClass::LessThan],
        "LogicalIndexMapApply" => &[EqClass::Copy],
        other => panic!("constructor {other} has NO shared-kernel mapping — extend kernels.rs"),
    }
}

fn main() {
    let _backend = CuteMetal::new();
    for (name, class) in KERNELS {
        println!("compiled {name:<16} {class:?}");
    }
    println!("\nall {} shared kernels compile and have pipelines", KERNELS.len());

    luminal::logical_ir::set_cute_lowering(true);
    let loaded = loaders::load(&cfg());
    luminal::logical_ir::set_cute_lowering(false);

    let have: BTreeSet<EqClass> = KERNELS.iter().map(|(_, c)| *c).collect();
    let mut ctors: BTreeSet<&str> = BTreeSet::new();
    for id in loaded.graph.logical.petgraph().node_indices() {
        ctors.insert(loaded.graph.logical.petgraph()[id].op.constructor());
    }
    for c in &ctors {
        for class in required(c) {
            assert!(
                have.contains(class),
                "model needs {class:?} for {c} but no shared kernel provides it"
            );
        }
    }
    println!(
        "census PASS: {} constructors in the loaded model, every one maps to shared kernels",
        ctors.len()
    );
}
