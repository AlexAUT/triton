#include "TritonAMDGPUTransforms/Passes.h"

#include "StreamPipeline.h"
#include "amd/lib/TritonAMDGPUToLLVM/SchedInstructions.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

#define DEBUG_TYPE "tritonamdgpu-stream-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

// Return true if the preconditions for pipelining the loop are met.
static bool checkPrecondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) { return !operand.getDefiningOp(); }))
    return false;

  auto hasInvalidOp = [forOp](Operation *op) {
    // Don't pipeline outer loops.
    if (op != forOp && isa<scf::ForOp, scf::WhileOp>(op))
      return WalkResult::interrupt();
    // Don't pipeline loops with barriers or asserts/prints.
    if (isa<gpu::BarrierOp, tt::AssertOp, tt::PrintOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  };
  return !forOp->walk(hasInvalidOp).wasInterrupted();
}

// Go through a single use chain to get the result of the target op after all
// unary ops - e.g., `convert_layout`, `fp_to_fp`, etc.
template <typename TargetOpType> Operation *passPrevUnaryOps(Value value) {
  auto getNextUnaryOps = [](Value value) -> Operation * {
    if (auto defOp = value.getDefiningOp()) {
      if ((defOp->getNumOperands() == 1) || llvm::dyn_cast<TargetOpType>(defOp))
        return defOp;
    }
    return nullptr;
  };

  auto unaryOp = getNextUnaryOps(value);
  while (unaryOp) {
    if (llvm::dyn_cast<TargetOpType>(unaryOp))
      return unaryOp;
    unaryOp = getNextUnaryOps(unaryOp->getOperand(0));
  }
  return nullptr;
}

// Annotate each `tt.LoadOp` instruction with its corresponding gemm operand
// index. Note, this is a part of the instruction scheduling routine. Currently,
// we support `forOp`s which contain only a single `tt.DotOp` in the bodies.
void labelLoadOpsForTritonDot(scf::ForOp forOp) {
  mlir::MLIRContext *ctx = forOp->getContext();
  if (auto dotOp = tt::getSingleDotOpIfExists(forOp)) {
    for (auto [opIdx, dotOperand] : llvm::enumerate(dotOp->getOperands())) {
      if (auto loadOp = passPrevUnaryOps<tt::LoadOp>(dotOperand)) {
        auto opIdxAttr = tt::amdgpu::OpIdxAttr::get(ctx, opIdx);
        loadOp->setAttr(tt::amdgpu::OpIdxAttr::getMnemonic(), opIdxAttr);
      }
    }
  }
}

struct PipelinePass : public TritonAMDGPUStreamPipelineBase<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int32_t _numStages, int32_t _globalPrefetch,
               int32_t _localPrefetch, bool _useAsyncCopy) {
    this->numStages = _numStages;

    this->globalPrefetch = _globalPrefetch;
    this->localPrefetch = _localPrefetch;

    this->useAsyncCopy = _useAsyncCopy;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // check numStages
    if (globalPrefetch < 0 || globalPrefetch >= numStages) {
      moduleOp.emitError("global prefetch control must be in [0, ")
          << numStages << "); " << globalPrefetch << " is out of range";
      return signalPassFailure();
    }

    if (localPrefetch < 0 || localPrefetch >= numStages) {
      moduleOp.emitError("local prefetch control must be in [0, ")
          << numStages << "); " << localPrefetch << " is out of range";
      return signalPassFailure();
    }

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      labelLoadOpsForTritonDot(forOp);
      // Bail out for loops with num_stage <= 1.
      if (tt::getNumStagesOrDefault(forOp, numStages) > 1)
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      if (!checkPrecondition(forOp))
        continue;
      StreamPipeliner sp(forOp, tt::getNumStagesOrDefault(forOp, numStages),
                         globalPrefetch, localPrefetch, useAsyncCopy);
      (void)sp.pipelineLoop();
    }

    if (useAsyncCopy) {
      llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
      moduleOp.walk([&](ttg::AsyncWaitOp waitOp) { waitOps.insert(waitOp); });
      tt::combineRedundantWaitOps(waitOps);
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUStreamPipelinePass(
    int numStages, int globalPrefetch, int localPrefetch, bool useAsyncCopy) {
  return std::make_unique<PipelinePass>(numStages, globalPrefetch,
                                        localPrefetch, useAsyncCopy);
}
