//! Search-selectable tiled matmul: the egglog match rule mints one producer
//! enode per tile config (MATMUL_CONFIGS), so Luminal's genome search chooses
//! the config by profiling. The kernel for each config is GENERATED from
//! cute-layout's algebra + cute_msl (register-tiled shared-memory GEMM).

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, matmul_tiled_codegen};
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulTiled {
    pub config: i64,
}

impl OpSlotNames for MatmulTiled {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatmulTiled {
    fn label(&self) -> &str {
        "MatmulTiledGeneric"
    }
}

impl Bufferizable for MatmulTiled {}

impl ToDps for MatmulTiled {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MatmulTiledDps {
            config: self.config,
        }))
    }
}

impl LayoutIrOp for MatmulTiled {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulTiledDps {
    pub config: i64,
}

impl OpSlotNames for MatmulTiledDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatmulTiledDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "MatmulTiledGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 2 // dest0 is write-only
    }
}

impl Bufferizable for MatmulTiledDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MatmulTiledDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for MatmulTiledDps {}

impl KernelOp for MatmulTiledDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        matmul_tiled_codegen(ctx, usize::try_from(self.config).unwrap_or(usize::MAX))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MatmulTiledMatcher;

impl OpMatcher for MatmulTiledMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpMatmulTiledGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        // Constructor args: (a_lt, b_lt, out_layout, config).
        &[("out_layout", 2)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MatmulTiled {
            config: site.child_i64(3),
        })
    }
}
