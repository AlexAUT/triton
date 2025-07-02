#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUREDUCEDATADUPLICATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUReduceDataDuplicationPass
    : public impl::TritonGPUReduceDataDuplicationBase<
          TritonGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
      if (!cvtNeedsSharedMemory(srcType, dstType))
        return;
      auto order = getOrderForMemory(srcType);
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());

      namespace ttg = triton::gpu;
      auto ctaLayout = ttg::getCTALayout(srcEncoding);
      auto srcTy = srcType;
      unsigned bitWidth = 16;
      auto dotOpEnc = dstDotOp;
      auto *ctx = cvtOp->getContext();
      unsigned innerD = ttg::getShapePerCTA(ctaLayout.getCTASplitNum(),
                                            srcTy.getShape())[order[0]];
      unsigned byteWidth = std::max(bitWidth / 8u, 1u);
      unsigned threadNumBytes = std::max(dotOpEnc.getKWidth() * byteWidth, 1u);
      auto sharedOrder = getOrderForMemory(srcTy);
      threadNumBytes = llvm::alignTo(
          threadNumBytes, std::max(4u, byteWidth)); // Assume 32-bit per bank
      unsigned paddingInElems = threadNumBytes / byteWidth;
      auto paddedAttr = ttg::PaddedSharedEncodingAttr::get(
          ctx, {{innerD, paddingInElems}}, sharedOrder, ctaLayout);

      auto tmpType = triton::gpu::MemDescType::get(
          dstType.getShape(), dstType.getElementType(), paddedAttr,
          // triton::gpu::SwizzledSharedEncodingAttr::get(
          //     mod.getContext(), dstDotOp, srcType.getShape(), order,
          //     triton::gpu::getCTALayout(srcEncoding),
          //     srcType.getElementType()),
          sharedMemorySpace);
      auto tmp = builder.create<triton::gpu::LocalAllocOp>(
          cvtOp.getLoc(), tmpType, cvtOp.getSrc());
      auto newConvert = builder.create<triton::gpu::LocalLoadOp>(cvtOp.getLoc(),
                                                                 dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
