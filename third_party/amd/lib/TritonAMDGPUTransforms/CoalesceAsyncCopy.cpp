#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = triton::gpu;

struct CoalesceAsyncCopySharedWrites
    : public OpRewritePattern<ttg::AsyncCopyGlobalToLocalOp> {
  CoalesceAsyncCopySharedWrites(triton::AMD::ISAFamily isaFamiliy,
                                MLIRContext *ctx)
      : OpRewritePattern(ctx), isaFamily{isaFamiliy} {}

  llvm::SmallVector<unsigned, 8>
  supportedAsyncLoadBitWidths(triton::AMD::ISAFamily isaFamily) const {
    switch (isaFamily) {
    case triton::AMD::ISAFamily::CDNA1:
    case triton::AMD::ISAFamily::CDNA2:
    case triton::AMD::ISAFamily::CDNA3:
      return {32};
    case triton::AMD::ISAFamily::CDNA4:
      return {32, 128};
    default:
      return {};
    }
  }

  // On gfx9 global and buffer loads directly to lds need to write coalesced
  // into LDS so we need to ensure that the src and dst layout result in
  // coalsced writes.
  // We can simply set the sizePerThread in the fastest dimension to the
  // supported load width and adjust threadsPerWarp to exhaust the whole fastest
  // dimension
  // FIXME: support rank > 2
  LogicalResult matchAndRewrite(ttg::AsyncCopyGlobalToLocalOp asyncCopy,
                                PatternRewriter &rewriter) const override {
    auto src = asyncCopy.getSrc();
    auto dst = asyncCopy.getResult();
    Value mask = asyncCopy.getMask();
    Value other = asyncCopy.getOther();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());

    auto srcEncoding = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!srcEncoding) {
      return rewriter.notifyMatchFailure(asyncCopy,
                                         "src encoding must be #blocked");
    }

    auto sharedEncoding =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    if (!sharedEncoding) {
      return rewriter.notifyMatchFailure(
          asyncCopy, "destination encoding must be SwizzledShared");
    }

    auto shape = dstTy.getShape();
    if (shape.size() > 2) {
      return rewriter.notifyMatchFailure(asyncCopy->getLoc(),
                                         " does only support 2D shapes");
    }

    auto order = sharedEncoding.getOrder();
    int warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(
        asyncCopy->getParentOfType<ModuleOp>());

    llvm::SmallVector<unsigned, 2> sizePerThread{1, 1};
    llvm::SmallVector<unsigned, 2> threadsPerWarp{1, 1};

    auto supportedBitWidths = supportedAsyncLoadBitWidths(isaFamily);
    unsigned elemBitWidth = dstTy.getElementTypeBitWidth();

    // Try all supported widths, we reverse to start from the larger sizes
    bool foundCoalescedConfig = false;
    for (unsigned bitsPerLoad : llvm::reverse(supportedBitWidths)) {
      unsigned elemBitWidth = dstTy.getElementTypeBitWidth();
      // Skip if the load width is not a multiple of element (covers smaller
      // than elemBitWidth)
      if ((bitsPerLoad % elemBitWidth) != 0)
        continue;

      // To ensure coalesced writes each threads holds a load width chunk of
      // data in the fastest dimension
      sizePerThread[order[0]] = bitsPerLoad / elemBitWidth;
      // We exhaust the fastest dimension and overlow into the second dim
      threadsPerWarp[order[0]] = std::min<unsigned>(
          warpSize, shape[order[0]] / sizePerThread[order[0]]);
      threadsPerWarp[order[1]] =
          std::max<unsigned>(1, warpSize / threadsPerWarp[order[0]]);

      foundCoalescedConfig = true;
      break;
    }
    if (!foundCoalescedConfig) {
      return rewriter.notifyMatchFailure(
          asyncCopy, "Failed to find a blocked layout configuration "
                     "resulting in coalesced writes");
    }

    // Return if we already use the found layout
    if (sizePerThread == srcEncoding.getSizePerThread() &&
        threadsPerWarp == srcEncoding.getThreadsPerWarp()) {
      return rewriter.notifyMatchFailure(asyncCopy, "already coalesced");
    }

    auto newLayout = ttg::BlockedEncodingAttr::get(
        asyncCopy->getContext(), sizePerThread, threadsPerWarp,
        ttg::getWarpsPerCTA(srcEncoding), ttg::getOrder(srcTy),
        ttg::getCTALayout(srcEncoding));

    auto convertLayout = [&rewriter](auto loc, Value old, auto newLayout) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newSrcTy = RankedTensorType::get(
          oldTy.getShape(), oldTy.getElementType(), newLayout);
      return rewriter.create<ttg::ConvertLayoutOp>(loc, newSrcTy, old);
    };

    auto loc = asyncCopy->getLoc();
    auto cvtSrc = convertLayout(loc, src, newLayout);
    if (mask)
      mask = convertLayout(loc, mask, newLayout);
    if (other)
      other = convertLayout(loc, other, newLayout);

    rewriter.modifyOpInPlace(asyncCopy, [&]() {
      asyncCopy.getSrcMutable().assign(cvtSrc);
      if (mask)
        asyncCopy.getMaskMutable().assign(mask);
      if (other)
        asyncCopy.getOtherMutable().assign(other);
    });

    return success();
  }

private:
  triton::AMD::ISAFamily isaFamily;
};

class TritonAMDGPUCoalesceAsyncCopyPass
    : public TritonAMDGPUCoalesceAsyncCopyBase<
          TritonAMDGPUCoalesceAsyncCopyPass> {
public:
  TritonAMDGPUCoalesceAsyncCopyPass(std::string_view archGenName)
      : isaFamily(triton::AMD::deduceISAFamily(archGenName)) {}

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    mlir::RewritePatternSet patterns(context);

    switch (isaFamily) {
    case triton::AMD::ISAFamily::CDNA1:
    case triton::AMD::ISAFamily::CDNA2:
    case triton::AMD::ISAFamily::CDNA3:
    case triton::AMD::ISAFamily::CDNA4:
      patterns.add<CoalesceAsyncCopySharedWrites>(isaFamily, context);
      break;
    default:
      break;
    }

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }

private:
  triton::AMD::ISAFamily isaFamily;
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUCoalesceAsyncCopy(const std::string archGenName) {
  return std::make_unique<TritonAMDGPUCoalesceAsyncCopyPass>(archGenName);
}
