#include "TritonAMDGPUToLLVM/TargetUtils.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = triton::gpu;

class TritonAMDGPUCoalesceAsyncCopyPass
    : public TritonAMDGPUCoalesceAsyncCopyBase<
          TritonAMDGPUCoalesceAsyncCopyPass> {
public:
  TritonAMDGPUCoalesceAsyncCopyPass(std::string_view archGenName)
      : isaFamily(triton::AMD::deduceISAFamily(archGenName)) {}

  void runOnOperation() override {
    SmallVector<ttg::AsyncCopyGlobalToLocalOp> asyncCopies;
    getOperation().walk([&](ttg::AsyncCopyGlobalToLocalOp asyncCopy) {
      asyncCopies.push_back(asyncCopy);
    });

    for (auto asyncCopy : asyncCopies) {
      switch (isaFamily) {
      case triton::AMD::ISAFamily::CDNA1:
      case triton::AMD::ISAFamily::CDNA2:
      case triton::AMD::ISAFamily::CDNA3:
      case triton::AMD::ISAFamily::CDNA4:
        if (coalesceWrites(asyncCopy).failed()) {
          signalPassFailure();
          return;
        }
      default:
        break;
      }
    }
  }

  llvm::SmallVector<unsigned, 8>
  supportedAsyncLoadBitWidths(triton::AMD::ISAFamily isaFamily) {
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

  // On gfx9 global.load.lds is limited to write coalesced into LDS so we
  // need to ensure that the src and dst layout result in coalsced writes.
  // We have to look at two distinct cases:
  //   1) If we do not swizzle into LDS, then we can simply set the
  //   sizePerThread in the fastest dimension to the supported load width
  //   and adjust threadsPerWarp to exhaust the whole fastest dimension
  //
  //   2) If we do swizzle into LDS TODO
  LogicalResult coalesceWrites(ttg::AsyncCopyGlobalToLocalOp asyncCopy) {
    auto src = asyncCopy.getSrc();
    auto dst = asyncCopy.getResult();
    auto srcTy = dyn_cast<ttg::TensorOrMemDesc>(src.getType());
    auto dstTy = dyn_cast<ttg::MemDescType>(dst.getType());
    if (!srcTy || !dstTy)
      return failure();

    auto sharedEncoding =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    if (!sharedEncoding)
      return failure();

    auto shape = dstTy.getShape();
    if (shape.size() > 2) {
      emitError(asyncCopy->getLoc()) << " does only support 2D shapes";
      signalPassFailure();
      return failure();
    }

    auto order = sharedEncoding.getOrder();
    // unsigned warpSize = product(ttg::getThreadsPerWarp(srcTy.getEncoding()));
    unsigned warpSize = 64; // FIXME: lookup warp size

    llvm::SmallVector<unsigned, 2> sizePerThread{1, 1};
    llvm::SmallVector<unsigned, 2> threadsPerWarp{1, 1};
    // TODO try all sizes
    auto supportedBitWidths = supportedAsyncLoadBitWidths(isaFamily);

    // Try all supported widths, we reverse to start from the larger sizes
    bool canCoalesce = false;
    for (unsigned bitsPerLoad : llvm::reverse(supportedBitWidths)) {
      unsigned elemBitWidth = dstTy.getElementTypeBitWidth();
      // Prevent selecting a smaller load size than our elem size
      if ((bitsPerLoad % elemBitWidth) != 0)
        continue;

      // To have coalesced writes each thread must only store data is the
      // fastest changing dimension
      sizePerThread[order[0]] =
          bitsPerLoad / dstTy.getElementType().getIntOrFloatBitWidth();
      // Same idea as above, we exhaust the fastest changing dimensions and span
      // over multiple elements of the second dimensions if needed
      threadsPerWarp[order[0]] = std::min<unsigned>(
          warpSize, shape[order[0]] / sizePerThread[order[0]]);
      threadsPerWarp[order[1]] =
          std::max<unsigned>(1, warpSize / threadsPerWarp[order[0]]);

      canCoalesce = true;
      break;
    }

    if (!canCoalesce) {
      emitError(asyncCopy->getLoc()) << "Failed to coalesce writes "
                                        "which is a hardware requirement";
      signalPassFailure();
      return failure();
    }

    auto srcEncoding = cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());

    // Return if we already use this configuration
    if (sizePerThread == srcEncoding.getSizePerThread() &&
        threadsPerWarp == srcEncoding.getThreadsPerWarp())
      return success();

    auto newLayout = ttg::BlockedEncodingAttr::get(
        asyncCopy->getContext(), sizePerThread, threadsPerWarp,
        ttg::getWarpsPerCTA(srcEncoding), ttg::getOrder(srcTy),
        ttg::getCTALayout(srcEncoding));

    RankedTensorType newSrcTy = RankedTensorType::get(
        srcTy.getShape(), srcTy.getElementType(), newLayout);
    IRRewriter builder(asyncCopy.getContext());
    builder.setInsertionPoint(asyncCopy);
    auto cvtSrc =
        builder.create<ttg::ConvertLayoutOp>(asyncCopy.getLoc(), newSrcTy, src);

    Value mask = asyncCopy.getMask();
    if (mask) {
      auto maskTy = dyn_cast<ttg::TensorOrMemDesc>(mask.getType());
      assert(maskTy);
      RankedTensorType newMaskTy = RankedTensorType::get(
          maskTy.getShape(), maskTy.getElementType(), newLayout);
      auto cvtMask = builder.create<ttg::ConvertLayoutOp>(asyncCopy->getLoc(),
                                                          newMaskTy, mask);
      mask = cvtMask;
    }

    Value other = asyncCopy.getOther();
    if (other) {
      auto otherTy = dyn_cast<ttg::TensorOrMemDesc>(other.getType());
      assert(otherTy);
      RankedTensorType newOtherTy = RankedTensorType::get(
          otherTy.getShape(), otherTy.getElementType(), newLayout);
      auto cvtOther = builder.create<ttg::ConvertLayoutOp>(asyncCopy->getLoc(),
                                                           newOtherTy, other);
      other = cvtOther;
    }

    builder.replaceOp(asyncCopy, builder.create<ttg::AsyncCopyGlobalToLocalOp>(
                                     asyncCopy.getLoc(), cvtSrc.getResult(),
                                     asyncCopy.getResult(), mask, other,
                                     asyncCopy.getCache(), asyncCopy.getEvict(),
                                     asyncCopy.getIsVolatile()));
    return success();
  }

private:
  triton::AMD::ISAFamily isaFamily;
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUCoalesceAsyncCopy(const std::string archGenName) {
  return std::make_unique<TritonAMDGPUCoalesceAsyncCopyPass>(archGenName);
}
