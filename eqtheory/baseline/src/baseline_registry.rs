use anyhow::{bail, Result};
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::dtype::PlanDtype;
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};
use luminal_metal::kernels::{layout_read_index, CodegenCtx, Coords, KernelOp, KernelSource};
use luminal_metal::symbolic::Expr;
use luminal_metal::{MetalOpInterface, RegisteredOp};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsqrtFunctional;

impl OpSlotNames for RsqrtFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RsqrtFunctional {
    fn label(&self) -> &str {
        "RsqrtFunctionalGeneric"
    }
}

impl Bufferizable for RsqrtFunctional {}

impl ToDps for RsqrtFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(RsqrtFunctionalDps))
    }
}

impl LayoutIrOp for RsqrtFunctional {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsqrtFunctionalDps;

impl OpSlotNames for RsqrtFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RsqrtFunctionalDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "RsqrtFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1
    }
}

impl Bufferizable for RsqrtFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for RsqrtFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for RsqrtFunctionalDps {}

impl KernelOp for RsqrtFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        if ctx.operand_dtypes[0] != PlanDtype::F32 || ctx.dest_dtypes[0] != PlanDtype::F32 {
            bail!("rsqrt fused kernel is F32 only");
        }
        let out_dims = &ctx.dest_dims[0];
        let n = out_dims
            .iter()
            .cloned()
            .reduce(|a, b| a * b)
            .unwrap_or_else(|| Expr::from(1usize));
        let layout = ctx.operand_layout(0);
        let (code, idx) =
            layout_read_index("a", layout, out_dims, Coords::FlatIndex { prefix: "c" })?;
        if !code.is_empty() || idx != "i" {
            bail!("rsqrt fused kernel requires a contiguous operand");
        }
        let source = format!(
            "kernel void k(device const float* a, device float* out, device const long* params, uint gid [[thread_position_in_grid]]) {{\n    const ulong n = {n};\n    ulong i = gid;\n    if (i < n) out[i] = 1.0f / sqrt(a[i]);\n}}"
        );
        Ok(vec![KernelSource::plain(source, n)])
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RsqrtFunctionalMatcher;

impl OpMatcher for RsqrtFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpRsqrtFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LayoutOpConstructors,
                text: include_str!("rsqrt_constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("rsqrt_match.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("layout", 1)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(RsqrtFunctional)
    }
}

pub fn shared_registry() -> Vec<RegisteredOp> {
    let mut reg = luminal_metal::metal_registry();
    reg.push(RegisteredOp::new(
        Box::new(RsqrtFunctionalMatcher),
        Box::new(RsqrtFunctional),
    ));
    reg
}
