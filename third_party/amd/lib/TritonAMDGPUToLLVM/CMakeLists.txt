add_triton_library(TritonAMDGPUToLLVM
    AsyncUtility.cpp
    AtomicRMWOpsEmitter.cpp
    AllocateSharedMemory.cpp
    BufferOpsEmitter.cpp
    ConvertLayoutOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/MFMA.cpp
    DotOpToLLVM/WMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    GCNAsmFormat.cpp
    TritonGPUToLLVM.cpp
    BuiltinFuncToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    TargetUtils.cpp
    OptimizeLDSUsage.cpp
    OptimizeLDSUtility.cpp
    SPMDOpToLLVM.cpp
    SchedInstructions.cpp
    UpcastMXFPToLLVM.cpp
    MembarUtility.cpp
    ScalarizePackedFOps.cpp

    DEPENDS
    TritonAMDGPUConversionPassIncGen
    LLVMIRIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonAMDGPUIR
    LLVMCore
    LLVMPasses
    LLVMSupport
)
