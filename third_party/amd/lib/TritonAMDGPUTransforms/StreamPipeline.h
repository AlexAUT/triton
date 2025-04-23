#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create stream operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop and epilogue.
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// Software pipelining generally works by anchoring on global load ops in the
// main loop and rotating the loop to schedule global load ops for future loop
// iterations together with compute for the current iteration. In this way, we
// can 1) issue memory operations earlier to hide the latency and 2) break the
// strong dependency inside on loop iteration to give backends flexibility to
// better interleave instructions for better instruction-level parallelism.
//
// This StreamPipeliner class creates the pipelining schedule and calls the
// PipelineExpander to rewrite the `scf.for` loop accordingly. A schedule
// consists of multiple stages, where ops from different stages can overlap
// executions because the dependencies are loop carried.
//
// The general flow of this process is:
//
// 1. The user provides a `num_stages` that specifies how many stages the
//    pipeline will have. The number of stages must be larger than the distance
//    from the first independent load to the compute in order to pipeline.
//    1.a. User may also specify `global_prefetch=<s>` to set the number of
//         stages between tt.load and ttg.local_store ops.
//    1.b. User may also specify `local_prefetch=<s>` to set the number of
//         stages between ttg.local_load and compute.
// 2. A schedule is created based on the distance between the global loads
//    in the first stages and the compute that uses the loaded values in the
//    last stage (num_stages - 1). Each operation will be clustered in the
//    order to best overlap with other operations (see details below in the
//    initSchedule method).
// 3. When the compute is a tt.dot, the scheduler will insert a shared
//    memory allocation between the global load and tt.dot. The ttg.local_store
//    will save the global load value to shared memory and the ttg.local_load
//    will load the relevant tiles for the tt.dot. These operations will be
//    scheduled according to various scheduling schemes outlined below in the
//    initSchedule method (see details there).
// 4. Finally the schedule will be passed to the PipelineExpander to rewrite
//    accordingly. The new implementation will consist of:
//    a. Prologue: containing the ramp-up of num_stages-1 stages for
//       iteratorions i=[0, num_stages-1).
//    b. New loop: ordered by cluster and iterated on each operation by
//       `i + (num_stages-op_stage)`.
//    c. Epilogue: ramp-down of the last `num_stages-1` iterations for the
//       ops in stages 1 to last_stage. This must consider that the loop
//       bounds may be shorter than num_stages. In this case, the epilogue
//       iterations must align with the prologue.
//
class StreamPipeliner {
  // Define categories of scheduling details per Operation types.
  // The StreamPipeliner schedules 5 types of operations:
  // 1. GLOBAL_LOAD: tt.load
  // 2. LOCAL_STORE: ttg.local_store (created by the StreamPipeliner)
  // 3. LOCAL_LOAD:  ttg.local_load (created by the StreamPipeliner)
  // 4. COMPUTE:     ops that use the loaded data
  // 5. ASYNC_WAIT:  ttg.async_wait (created by the StreamPipeliner)
  enum SchedType {
    SCHED_GLOBAL_LOAD,
    SCHED_LOCAL_STORE,
    SCHED_LOCAL_LOAD,
    SCHED_COMPUTE,
    // SCHED_ASYNC_WAIT,
    SCHED_SIZE
  };

public:
  StreamPipeliner(scf::ForOp _forOp, int _numStages, int _globalPrefetch,
                  int _localPrefetch, bool _useAsyncCopy);
  LogicalResult pipelineLoop();

private:
  LogicalResult initSchedule(int maxIndirectionLevel);

  void computeLoadOpsToIndirectionLevelAndUse();
  void assignMemoryLayouts();
  LogicalResult scheduleLoads(DenseSet<Operation *> &rootUsers);
  void scheduleDependencies();
  void scheduleDistanceOneDependencies();
  void scheduleRemainingToLastStage();

  LogicalResult preprocessLoopAndBuildSchedule();

  Value createAlloc(Operation *loadOp,
                    ttg::SwizzledSharedEncodingAttr sharedEnc);
  bool createAsyncCopy(tt::LoadOp loadOp, Value alloc, Value extractIdx);
  void createStreamCopy(tt::LoadOp loadOp, Value alloc, Value extractIdx);
  void createStreamOps();

  void scheduleOp(Operation *op, SchedType type, int stage = -1) {
    if (stage < 0)
      stage = stages[type];
    schedule.insert(op, stage, clusters[type]);
  }

private:
  // Data members
  scf::ForOp forOp;

  // User settings
  int numStages;

  // Computed number of buffers
  int numBuffers;

  // Directly store to shared memory with AsyncCopy when pipelining tt.loads
  bool useAsyncCopy;

  // Stage for each SchedType Op
  int stages[SCHED_SIZE];
  // Cluster for each SchedType Op
  std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> clusters;
  std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> clusterVec;

  // Scheduling clusters
  tt::CoarseSchedule schedule;

  // Mapping and indirection level for each `tt.load` to its use.
  SmallVector<std::tuple<Operation *, int, Operation *>> loadOpToIndLevelAndUse;

  struct LoadInfo {
    // Shared layout is used for loads feeding into dot ops.
    ttg::SwizzledSharedEncodingAttr sharedEncoding = nullptr;
    // The distance of this load's stage to its use' stage.
    int distToUse = 0;
    bool usedByDot = false;
    bool isAsync = false;
  };

  // Mapping for each pipelined load to scheduling details.
  llvm::MapVector<Operation *, LoadInfo> loadToInfo;

  // Lookup alignment/contiguity mappings for the current module.
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis;

  // Capture list of new shared memory buffers.
  SmallVector<Value> sharedMemAllocs;

  // Pipelining options for the PipelineExpander
  tt::PipeliningOption options;
};
