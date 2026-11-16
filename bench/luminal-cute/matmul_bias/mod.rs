//! Fused matmul + bias-add (GEMM epilogue fusion, Step 2 of the Luminal-CuTe
//! investigation). One kernel computes `out[m,n] = sum_k A[m,k]*B[n,k] +
//! bias[m,n]`, applying the bias in the epilogue before the store so the M*N
//! matmul intermediate never round-trips global memory.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, matmul_bias_codegen};
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulBias {
    pub axis: i64,
}

impl OpSlotNames for MatmulBias {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "bias".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatmulBias {
    fn label(&self) -> &str {
        "MatmulBiasGeneric"
    }
}

impl Bufferizable for MatmulBias {}

impl ToDps for MatmulBias {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MatmulBiasDps { axis: self.axis }))
    }
}

impl LayoutIrOp for MatmulBias {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulBiasDps {
    pub axis: i64,
}

impl OpSlotNames for MatmulBiasDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "bias".to_string(),
            3 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatmulBiasDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "MatmulBiasGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand < 3 // dest0 (operand 3) is write-only
    }
}

impl Bufferizable for MatmulBiasDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 3,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MatmulBiasDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for MatmulBiasDps {}

impl KernelOp for MatmulBiasDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        matmul_bias_codegen(ctx)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MatmulBiasMatcher;

impl OpMatcher for MatmulBiasMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpMatmulBiasGeneric"
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
        // Constructor args: (a_lt, b_lt, bias_lt, axis, out_layout).
        &[("axis", 3), ("out_layout", 4)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MatmulBias {
            axis: site.child_i64(3),
        })
    }
}
