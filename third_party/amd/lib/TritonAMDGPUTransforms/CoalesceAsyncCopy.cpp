#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = triton::gpu;

// On gfx9 global and buffer loads directly to shared memory need to write
// coalesced. This pattern converts the layout of the src, mask and other to
// ensure the owned data per thread is contigious and does no exceed the
// supported load vector size. The swizzle pattern is ignored here and is
// handled when lowering to LLVMIR
struct CoalesceAsyncCopyWrites
    : public OpRewritePattern<ttg::AsyncCopyGlobalToLocalOp> {
  CoalesceAsyncCopyWrites(const triton::AMD::TargetInfo &targetInfo,
                          const DenseMap<ttg::AsyncCopyGlobalToLocalOp,
                                         unsigned> &asyncCopyContiguity,
                          MLIRContext *ctx)
      : OpRewritePattern(ctx), targetInfo{targetInfo},
        asyncCopyContiguity{std::move(asyncCopyContiguity)} {}

  LogicalResult matchAndRewrite(ttg::AsyncCopyGlobalToLocalOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto src = copyOp.getSrc();
    auto dst = copyOp.getResult();
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());

    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(copyOp,
                                         "src encoding must be #blocked");

    auto blockedLL = triton::gpu::toLinearLayout(srcTy.getShape(), blockedEnc);

    MLIRContext *ctx = getContext();
    auto order = blockedEnc.getOrder();

    auto loc = copyOp->getLoc();
    // Convert layout of src, mask and other to new encoding
    auto convertLayout = [&rewriter](auto loc, Value old, auto newEnc) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newSrcTy = RankedTensorType::get(
          oldTy.getShape(), oldTy.getElementType(), newEnc);
      return rewriter.create<ttg::ConvertLayoutOp>(loc, newSrcTy, old);
    };

    auto paddedEnc =
        dyn_cast<ttg::PaddedSharedEncodingAttr>(dstTy.getEncoding());

    if (paddedEnc) {

      llvm::outs() << "Order: " << order[0] << ", " << order[1] << "\n";
      llvm::outs() << "RegBlocked:\n" << blockedLL << "\n";
      // LinearLayout reg = triton::identityStandardND(
      //     str_attr("register"), blockedEnc.getSizePerThread(), order);
      // llvm::outs() << "Reg:\n" << reg << "\n";
      auto standardOutDims = standardOutDimNames(ctx, srcTy.getRank());
      StringAttr kRegister = StringAttr::get(ctx, "register");
      StringAttr kLane = StringAttr::get(ctx, "lane");
      StringAttr kWarp = StringAttr::get(ctx, "warp");
      StringAttr kBlock = StringAttr::get(ctx, "block");

      unsigned contigDimSize = srcTy.getShape()[paddedEnc.getOrder()[0]];
      unsigned nonContigDimSize = srcTy.getShape()[paddedEnc.getOrder()[1]];
      llvm::outs() << "Contig size: " << contigDimSize << "\n";

      std::vector<std::vector<int>> regBases;
      std::vector<std::vector<int>> laneBases;
      std::vector<std::vector<int>> warpBases;
      if (contigDimSize == 256) {
        if (nonContigDimSize == 64) {
          regBases = {{1, 0}, {2, 0}, {4, 0}, {0, 8}, {128, 0}};
          laneBases = {{8, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 16}, {0, 32}};
          warpBases = {{0, 1}, {0, 2}, {0, 4}};
        } else {
          assert(false);
        }
      }
      if (contigDimSize == 128) {
        if (nonContigDimSize == 64) {
          regBases = {{1, 0}, {2, 0}, {4, 0}, {0, 1}};
          laneBases = {{8, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 16}, {0, 32}};
          warpBases = {{0, 2}, {0, 4}, {0, 8}};
        } else if (nonContigDimSize == 256) {
          regBases = {{1, 0}, {2, 0}, {4, 0}, {0, 1}, {0, 2}, {0, 4}};
          laneBases = {{8, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 16}, {0, 32}};
          warpBases = {{0, 8}, {0, 64}, {0, 128}};
        } else {
          assert(false);
        }
      }
      if (contigDimSize == 64) {
        if (nonContigDimSize == 256) {
          regBases = {{1, 0}, {2, 0}, {4, 0}, {0, 8}, {0, 128}};
          laneBases = {{8, 0}, {16, 0}, {32, 0}, {0, 16}, {0, 32}, {0, 64}};
          warpBases = {{0, 1}, {0, 2}, {0, 4}};
        } else {
          assert(false);
        }
      }

      auto transposeBases = [](std::vector<std::vector<int>> &vec) {
        for (auto &p : vec)
          std::swap(p[0], p[1]);
      };

      if (order[0] != 0) {
        transposeBases(regBases);
        transposeBases(laneBases);
        transposeBases(warpBases);
      }

      LinearLayout paddedLayout({{kRegister, regBases},
                                 {kLane, laneBases},
                                 {kWarp, warpBases},
                                 {kBlock, {}}},
                                {standardOutDims[0], standardOutDims[1]});

      auto llEnc = ttg::LinearEncodingAttr::get(ctx, paddedLayout);
      llvm::outs() << "Padding layout: " << paddedLayout << "\n";
      llvm::outs() << "Encoding: " << llEnc << "\n";
      auto cvtLL = convertLayout(loc, src, llEnc);
      if (mask)
        mask = convertLayout(loc, mask, llEnc);
      if (other)
        other = convertLayout(loc, other, llEnc);

      rewriter.modifyOpInPlace(copyOp, [&]() {
        copyOp.getSrcMutable().assign(cvtLL);
        if (mask)
          copyOp.getMaskMutable().assign(mask);
        if (other)
          copyOp.getOtherMutable().assign(other);
      });
      return success();
    }

    auto sharedEnc =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    if (!sharedEnc)
      return rewriter.notifyMatchFailure(
          copyOp, "destination encoding must be #SwizzledShared");

    // We start from the precomputed contiguity we got from AxisAnalysis.
    unsigned loadContig = 0;
    if (auto it = asyncCopyContiguity.find(copyOp);
        it != asyncCopyContiguity.end())
      loadContig = it->second;
    else
      return copyOp->emitError()
             << "No contiguity information about the copy op";
    assert(loadContig > 0);

    // Further restrict the contiguity based on the contiguity of the src to dst
    // layout e.g. if the order of the blocked and shared encoding is different
    // we can only load one element at a time or if the shared encoding is
    // swizzled we cannot exceed the vector size of the swizzling pattern
    LinearLayout regLayout =
        triton::gpu::toLinearLayout(srcTy.getShape(), blockedEnc);
    LinearLayout sharedLayout =
        triton::gpu::toLinearLayout(srcTy.getShape(), sharedEnc);
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);
    loadContig = std::min<unsigned>(loadContig,
                                    regToSharedLayout.getNumConsecutiveInOut());

    // Select the largest supported load width equal or smaller than loadContig
    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    while (loadContig > 0 && !targetInfo.supportsDirectToLdsLoadBitWidth(
                                 loadContig * elemBitWidth)) {
      loadContig /= 2;
    }

    if (loadContig == 0) {
      return rewriter.notifyMatchFailure(
          copyOp, "could not find layout config to create coalesced writes");
    }

    // Do not rewrite if we already use the correct contiguity (could be from a
    // previous rewrite)
    auto contigPerThread = ttg::getContigPerThread(srcTy);
    auto blockedContig = contigPerThread[blockedEnc.getOrder()[0]];
    if (blockedContig == loadContig) {
      return rewriter.notifyMatchFailure(copyOp,
                                         "already using the correct layout");
    }

    // Get new blocked encoding with loadContig as sizePerThread in the fastest
    // dim
    assert(blockedContig >= loadContig);
    contigPerThread[blockedEnc.getOrder()[0]] = loadContig;
    int numWarps = triton::gpu::lookupNumWarps(copyOp);
    auto mod = copyOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    auto newBlockEnc = BlockedEncodingAttr::get(
        copyOp.getContext(), srcTy.getShape(), contigPerThread,
        blockedEnc.getOrder(), numWarps, threadsPerWarp,
        blockedEnc.getCTALayout());

    Value cvtSrc = convertLayout(loc, src, newBlockEnc);

    if (mask)
      mask = convertLayout(loc, mask, newBlockEnc);
    if (other)
      other = convertLayout(loc, other, newBlockEnc);

    rewriter.modifyOpInPlace(copyOp, [&]() {
      copyOp.getSrcMutable().assign(cvtSrc);
      if (mask)
        copyOp.getMaskMutable().assign(mask);
      if (other)
        copyOp.getOtherMutable().assign(other);
    });
    return success();
  }

private:
  const triton::AMD::TargetInfo &targetInfo;
  const DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> &asyncCopyContiguity;
};

class TritonAMDGPUCoalesceAsyncCopyPass
    : public TritonAMDGPUCoalesceAsyncCopyBase<
          TritonAMDGPUCoalesceAsyncCopyPass> {
public:
  TritonAMDGPUCoalesceAsyncCopyPass(StringRef archGenName) {
    this->archGenerationName = archGenName.str();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    triton::AMD::TargetInfo targetInfo(archGenerationName);

    mlir::RewritePatternSet patterns(context);

    switch (targetInfo.getISAFamily()) {
    case triton::AMD::ISAFamily::CDNA1:
    case triton::AMD::ISAFamily::CDNA2:
    case triton::AMD::ISAFamily::CDNA3:
    case triton::AMD::ISAFamily::CDNA4: {
      break;
    }
    default:
      return;
    }

    // Precompute the contiguity of all AsyncCopy ops based on the src and
    // mask contiguity/alignment to avoid rebuilding ModuleAxisInfoAnalysis
    // after every IR change.
    triton::ModuleAxisInfoAnalysis axisAnalysis(m);
    DenseMap<ttg::AsyncCopyGlobalToLocalOp, unsigned> asyncCopyContiguity;
    m->walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
      unsigned contiguity =
          mlir::LLVM::AMD::getContiguity(copyOp.getSrc(), axisAnalysis);
      if (auto mask = copyOp.getMask()) {
        contiguity =
            std::min<unsigned>(contiguity, axisAnalysis.getMaskAlignment(mask));
      }
      asyncCopyContiguity.insert({copyOp, contiguity});
    });
    patterns.add<CoalesceAsyncCopyWrites>(targetInfo, asyncCopyContiguity,
                                          context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUCoalesceAsyncCopyPass(std::string archGenName) {
  return std::make_unique<TritonAMDGPUCoalesceAsyncCopyPass>(
      std::move(archGenName));
}
