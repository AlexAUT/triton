#include "Utility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace tt = mlir::triton;
using mlir::triton::ModuleAxisInfoAnalysis;
using mlir::triton::AMD::DppCtrl;
using mlir::triton::AMD::ISAFamily;
using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;

namespace {
enum class ShflKind : uint32_t {
  bfly = 0,
  up = 1,
  down = 2,
  idx = 3,
};

std::string getTypeString(Type ty) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  ty.print(rso);
  rso.flush();
  return str;
}

std::string mangleFunc(std::string name, Type type) {
  auto funcType = dyn_cast<LLVM::LLVMFunctionType>(type);
  assert(funcType && "Expecting an LLVMFunctionType");
  std::string mangled = name + "_";
  auto retTy = funcType.getReturnType();
  mangled += getTypeString(retTy) + "_";
  auto params = funcType.getParams();
  for (auto paramType : params) {
    mangled += getTypeString(paramType) + "_";
  }
  return mangled;
}

// Utility function to create a constant vector mask of length `vecSize` with
// the same `pred` value
Value createVectorMaskFromPredicate(RewriterBase &rewriter, Location loc,
                                    Value pred, int64_t vecSize) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto vecMaskTy = LLVM::getVectorType(rewriter.getI1Type(), vecSize);
  Value maskVal = b.undef(vecMaskTy);
  for (size_t s = 0; s < vecSize; ++s) {
    Value indexVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(s));
    maskVal = b.insert_element(vecMaskTy, maskVal, pred, indexVal);
  }
  return maskVal;
}

// Utility function to get the number of elements of a vector or a scalar
int64_t getNumElements(Type ty) {
  if (auto vecType = dyn_cast<VectorType>(ty))
    return vecType.getNumElements();
  return 1;
}

// Utility function to cast the given scalar or vector type to a vector type
Type castToVectorType(Type ty) {
  if (isa<VectorType>(ty))
    return ty;
  return LLVM::getVectorType(ty, 1);
}

} // namespace

namespace mlir::LLVM::AMD {
static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter,
                               ISAFamily isaFamily, Value val, Value i,
                               int strideInt, ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // On AMD, the ds_swizzle_b32 and ds_permute_b32 instructions work on
  // 32bit/dwords so we need promote to 32 here.
  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.sext(i32_ty, val);

    val = shuffleCommonImpl(loc, rewriter, isaFamily, val, i, strideInt, mode,
                            clamp);

    if (bits < 32)
      val = b.trunc(int_ty(bits), val);
    if (!valType.isIntOrIndex())
      val = b.bitcast(val, valType);
    return val;
  }

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl(loc, rewriter, isaFamily, val0, i, strideInt, mode,
                             clamp);
    val1 = shuffleCommonImpl(loc, rewriter, isaFamily, val1, i, strideInt, mode,
                             clamp);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Value threadId = getThreadId(rewriter, loc);

  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = b.i32_val(iWarpSize);
  Value laneId = b.urem(threadId, warpSize);
  auto bpermute = [&](Value lane) {
    // Multiple lineId by 4. (More on permute instruction semantics:
    // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf#page=180
    Value byteOffset = b.i32_val(2);
    Value permuteAddr = b.shl(lane, byteOffset);
    return rewriter.create<ROCDL::DsBpermuteOp>(loc, valType, permuteAddr, val);
  };

  switch (mode) {
  case ShflKind::bfly:
    if (strideInt > 16) {
      Value stride = b.i32_val(32);
      Value lineId = b.xor_(threadId, stride);
      return bpermute(lineId);
    } else if (strideInt == 16) {
      if (isRDNA(isaFamily)) {
        // Lane i in the upper 16 lanes reads the value from lane i in the lower
        // 16 lanes and vice versa.
        Value select_lo = b.i32_val(0x76543210);
        Value select_hi = b.i32_val(0xfedcba98);
        return rewriter.create<ROCDL::PermlaneX16Op>(
            loc, valType, val, val, select_lo, select_hi, true, false);
      } else {
        Value offset = b.i32_val(0x401F);
        return rewriter.create<ROCDL::DsSwizzleOp>(loc, valType, val, offset);
      }
    } else {
      if (!llvm::is_contained({ISAFamily::CDNA2, ISAFamily::CDNA3,
                               ISAFamily::CDNA4, ISAFamily::RDNA3},
                              isaFamily)) {
        // DPP is only supported for CDNA2/CDNA3/CDNA4/RDNA3 right now, so we
        // fallback to ds_swizzle for other architectures.
        //
        // This map facilates the butterfly shuffle pattern for a stride less
        // than 16. The pattern stride is the key of the map.
        DenseMap<short, unsigned int> masks{
            {16, 0x401F}, {8, 0x201F}, {4, 0x101F}, {2, 0x081F}, {1, 0x041F}};
        Value offset = b.i32_val(masks[strideInt]);
        return rewriter.create<ROCDL::DsSwizzleOp>(loc, valType, val, offset);
      }

      auto createDppOpWithoutBoundCtrl = [&](Value &old, Value &src,
                                             uint32_t dppCtrl, uint32_t rowMask,
                                             uint32_t bankMask) {
        return rewriter.create<ROCDL::DPPUpdateOp>(
            loc, valType, old, src, rewriter.getI32IntegerAttr(dppCtrl),
            rewriter.getI32IntegerAttr(rowMask),
            rewriter.getI32IntegerAttr(bankMask), rewriter.getBoolAttr(false));
      };

      const int allRows = 0xf;
      const int allBanks = 0xf;

      switch (strideInt) {
      case 1: {
        // quad_perm: 1, 0, 3, 2
        uint32_t dppCtrl = static_cast<uint32_t>(DppCtrl::QUAD_PERM_FIRST);
        std::array<uint32_t, 4> mask = {1, 0, 3, 2};
        for (int i = 0; i < mask.size(); i++) {
          dppCtrl |= mask[i] << (i * 2);
        }
        return createDppOpWithoutBoundCtrl(val, val, dppCtrl, allRows,
                                           allBanks);
      }
      case 2: {
        // quad_perm: 2, 3, 0, 1
        uint32_t dppCtrl = static_cast<uint32_t>(DppCtrl::QUAD_PERM_FIRST);
        std::array<uint32_t, 4> mask = {2, 3, 0, 1};
        for (int i = 0; i < mask.size(); i++) {
          dppCtrl |= mask[i] << (i * 2);
        }
        return createDppOpWithoutBoundCtrl(val, val, dppCtrl, allRows,
                                           allBanks);
      }
      case 4: {
        // row_shr:4 bank_mask: 0xa
        auto ret = createDppOpWithoutBoundCtrl(
                       val, val, 4 + static_cast<uint32_t>(DppCtrl::ROW_SHR0),
                       allRows, 0xa)
                       .getRes();

        // row_shl:4 bank_mask: 0x5
        return createDppOpWithoutBoundCtrl(
            ret, val, 4 + static_cast<uint32_t>(DppCtrl::ROW_SHL0), allRows,
            0x5);
      }
      case 8: {
        // row_shr:8 bank_mask: 0xc
        auto ret = createDppOpWithoutBoundCtrl(
                       val, val, 8 + static_cast<uint32_t>(DppCtrl::ROW_SHR0),
                       allRows, 0xc)
                       .getRes();

        // row_shl:8 bank_mask: 0x3
        return createDppOpWithoutBoundCtrl(
            ret, val, 8 + static_cast<uint32_t>(DppCtrl::ROW_SHL0), allRows,
            0x3);
      }
      default:
        assert(false &&
               "bfly shfl with stride >= 16 should not be handled by dpp.");
      }
    }
    break;
  case ShflKind::up: {
    Value mask = b.icmp_slt(laneId, i);
    Value delta = b.sub(laneId, i);
    Value index = b.select(mask, laneId, delta);
    return bpermute(index);
  }
  case ShflKind::idx:
    return bpermute(i);
  default:
    assert(false && "Unsupported ShflKind");
    break;
  }
  return Value();
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter,
                           ISAFamily isaFamily, Value val, Value i,
                           int strideInt, ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, isaFamily, val, i, strideInt,
                                   mode, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, isaFamily, val, b.i32_val(i), i,
                       ShflKind::bfly, b.i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, isaFamily, val, b.i32_val(i), i,
                       ShflKind::up, b.i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i), isaFamily);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, isaFamily, val, i, 0, ShflKind::idx,
                       b.i32_val(0x1f));
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis) {
  Value blockId =
      rewriter.create<::mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension(axis));
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, triton::CacheModifier cm,
             bool forceNoAliasAsyncLoads) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Type funcType = getFunctionType(elemTy, ValueRange({ptr, pred, falseVal}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto getLoadNameRaw = [](triton::CacheModifier cm) {
    switch (cm) {
    case triton::CacheModifier::CA:
      return predicatedLoadCA;
    case triton::CacheModifier::CG:
      return predicatedLoadCG;
    case triton::CacheModifier::CV:
      return predicatedLoadCV;
    default:
      // Do not fail in compile time in the case of unsupported modifier.
      // Just apply default config.
      return predicatedLoad;
    }
  };
  std::string funcName = getLoadNameRaw(cm);
  if (forceNoAliasAsyncLoads)
    funcName += noAliasAsyncLoads;

  auto mangledName = mangleFunc(funcName, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, mangledName, funcType);
  return LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                ValueRange({ptr, pred, falseVal}))
      .getResult();
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::CacheModifier cm,
             bool forceNoAliasAsyncLoads) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto ctx = ptr.getContext();
  Type funcType = getFunctionType(void_ty(ctx), ValueRange({ptr, val, pred}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();

  auto getStoreNameWithCacheMod = [](triton::CacheModifier cm) {
    switch (cm) {
    case triton::CacheModifier::WT:
      return predicatedStoreWT;
    case triton::CacheModifier::CG:
      return predicatedStoreCG;
    case triton::CacheModifier::CS:
      return predicatedStoreCS;
    default:
      // Do not fail in compile time in the case of unsupported modifier.
      // Just apply default config.
      return predicatedStore;
    }
  };
  std::string funcName = getStoreNameWithCacheMod(cm);
  if (forceNoAliasAsyncLoads)
    funcName += noAliasAsyncLoads;

  auto mangledName = mangleFunc(funcName, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, mangledName, funcType);
  LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange({ptr, val, pred}));
}

static bool isPredicatedLoadCA(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoadCA);
}

static bool isPredicatedLoadCG(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoadCG);
}

static bool isPredicatedLoadCV(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoadCV);
}

static bool isPredicatedStoreCS(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(
      mlir::LLVM::AMD::predicatedStoreCS);
}

static bool isPredicatedStoreCG(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(
      mlir::LLVM::AMD::predicatedStoreCG);
}

static bool isPredicatedStoreWT(LLVM::CallOp callOp) {
  return callOp.getCallee().value().contains(
      mlir::LLVM::AMD::predicatedStoreWT);
}

// Utility function that returns flags <volatile, nontemporal> for a predicated
// Load or Store
// ---------------------------------
// Op   | cm  | volatile | NT
// -----+-----+---------------------
// Load | .ca |   F      | F
//      | .cg |   F      | T
//      | .cs |   F      | T
//      | .cv |   T      | X
// -----+-----+----------+---------
// Store| .wb |   F      | F
//      | .cg |   F      | F
//      | .cs |   F      | T
//      | .wt |   T      | X
// -----+-----+----------+---------
std::pair<bool, bool>
getCacheModifierFlagsForPredicatedCall(LLVM::CallOp callOp) {
  if (isPredicatedLoadCA(callOp))
    return std::make_pair(false, false);
  if (isPredicatedLoadCG(callOp))
    return std::make_pair(false, true);
  if (isPredicatedLoadCV(callOp))
    return std::make_pair(true, true);

  if (isPredicatedStoreCG(callOp))
    return std::make_pair(false, false);
  if (isPredicatedStoreCS(callOp))
    return std::make_pair(false, true);
  if (isPredicatedStoreWT(callOp))
    return std::make_pair(true, true);
  // unsupported modifier
  return std::make_pair(false, false);
}

// Create the auxiliary/cachepolicy value of ROCDL::RawPtrBufferLoad/StoreOp
//   gfx942 and gfx950: bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
// Vector Memory instructions (Flat, Global, Scratch, and Buffer) have 3
// bits to control scope and cacheability:
// - SC[1:0] System Cache level: 0=wave, 1=group, 2=device, 3=system
// - NT Non-Temporal: 0=expect temporal reuse; 1=do not expect temporal reuse
//
// -------+-----+-----+-----+----+--
// Op     | cm  | SC1 | SC0 | NT |
// -------+-----+-----+-----+----+--
// Load   | .ca |  0  |  0  | 0  |
//        | .cg |  0  |  1  | 1  |
//        | .cs |  0  |  1  | 1  |
//        | .cv |  1  |  1  | x  |
// -------+-----+-----+-----+----+--
// Store  | .wb |  0  |  0  | 0  |
//        | .cg |  0  |  0  | 0  |
//        | .cs |  0  |  1  | 1  |
//        | .wt |  1  |  1  | x  |
// -------+-----+-----+-----+----+--
// Atomic | N/A |  0  |  1  | x  | Setting sc0 returns the pre-op value
//        | N/A |  1  |  0  | x  | Setting sc1 performs a system-scope atomic
// -------+-----+-----+-----+----+--
static int32_t
getCtrlBitsForCacheModifierOnGFX_942_950(triton::CacheModifier cm,
                                         bool isLoad) {
  const int sc0Bit = 0b1, ntBit = 0b10, sc1Bit = 0b10000;
  int32_t aux = 0;
  switch (cm) {
  case triton::CacheModifier::CA:
    aux = 0;
    break;
  case triton::CacheModifier::CG:
    if (isLoad)
      aux |= sc0Bit | ntBit;
    break;
  case triton::CacheModifier::CS:
    aux |= sc0Bit | ntBit;
    break;
  case triton::CacheModifier::CV:
    assert(isLoad);
    aux |= sc0Bit | sc1Bit;
    break;
  case triton::CacheModifier::WB:
    assert(!isLoad);
    aux = 0;
    break;
  case triton::CacheModifier::WT:
    assert(!isLoad);
    aux |= sc0Bit | sc1Bit;
    break;
  default:
    aux = 0;
  }
  return aux;
}

int32_t getCtrlBitsForBufferAtomicsOnGFX_942_950(bool setSC0, bool setSC1,
                                                 bool setNT) {
  const int sc0Bit = 0b1, ntBit = 0b10, sc1Bit = 0b10000;
  int32_t aux = 0;
  if (setSC0)
    aux |= sc0Bit;
  if (setSC1)
    aux |= sc1Bit;
  if (setNT)
    aux |= ntBit;
  return aux;
}

static int32_t getDefaultCtrlBitsForCacheModifier(triton::CacheModifier cm) {
  return 0;
}

// Cache modifiers changes how data is managed in the GPU's cache hierarchy:
// .ca: cache at all levels with LRU policy
// .cg: cache at L2, can use .ca or .cs
// .cs: cache streaming, use data once
// .cv: don't cache and fetch again
// .wb: write-back, writes back data at all cache levels
// .wt: write-through, write data directly to system memory
int32_t getCtrlBitsForCacheModifierOnTarget(
    triton::CacheModifier cm, bool isLoad,
    const mlir::triton::AMD::TargetInfo &targetInfo) {
  switch (targetInfo.getGPUKind()) {
  case llvm::AMDGPU::GK_GFX942:
  case llvm::AMDGPU::GK_GFX950:
    return getCtrlBitsForCacheModifierOnGFX_942_950(cm, isLoad);
  default:
    return getDefaultCtrlBitsForCacheModifier(cm);
  }
}

Value cvtFp32ToFp16RTNE_oneValue(Location loc, RewriterBase &rewriter,
                                 const Value &v) {
  LLVM::RoundingMode rm = LLVM::RoundingMode::NearestTiesToEven;
  return rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v);
}

Type getPointerTypeWithShape(Value basePtr, Value offset) {
  Type basePtrType = basePtr.getType();
  auto offsetType = cast<RankedTensorType>(offset.getType());
  return offsetType.cloneWith(std::nullopt, basePtrType);
}

unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  return axisAnalysisPass.getContiguity(ptr);
}

unsigned getContiguity(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass) {

  Type type = getPointerTypeWithShape(ptr, offset);
  RankedTensorType tensorTy = cast<RankedTensorType>(type);

  // To compute the contiguity of the scalar/warp-uniform ptr and offset pair we
  // need to look at the contiguity of the offsets and the alignment of the ptr
  auto elemNumBits = triton::getPointeeBitWidth(tensorTy);
  auto contiguity = axisAnalysisPass.getContiguity(offset, elemNumBits);

  // To get the alignment of the scalar ptr we need to look at the divisibility
  auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
  auto maxMultipleBytes = axisInfo->getDivisibility(0);
  auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
  auto align = std::max<unsigned>(maxMultipleBytes / elemNumBytes, 1);

  // FIXME (Alex): this should not be needed anymore because it's done inside
  // getContiguity, but we have an order issues with LL, so we keep this
  // until the LL order issue is fixed
  auto linearLayout = triton::gpu::toLinearLayout(tensorTy);
  auto llAttr =
      triton::gpu::LinearEncodingAttr::get(tensorTy.getContext(), linearLayout);
  auto order = triton::gpu::getOrder(tensorTy);
  auto contigPerThread = llAttr.getContigPerThread();
  assert(order[0] < contigPerThread.size() &&
         "Unexpected contigPerThread size");
  contiguity = std::min(contiguity, contigPerThread[order[0]]);

  // Final contiguity is a min of the offset contiguity and pointer alignment
  return std::min(align, contiguity);
}

unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  auto contiguity = getContiguity(ptr, axisAnalysisPass);
  auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}

unsigned getVectorSize(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto contiguity = getContiguity(ptr, offset, axisAnalysisPass);
  auto pointeeBitWidth = triton::getPointeeBitWidth(ptr.getType());
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}

Type scaleDotElemTypeToMLIRType(MLIRContext *ctx, triton::ScaleDotElemType t) {
  switch (t) {
  case triton::ScaleDotElemType::FP16:
    return Float16Type::get(ctx);
  case triton::ScaleDotElemType::BF16:
    return BFloat16Type::get(ctx);
  case triton::ScaleDotElemType::E4M3:
    return Float8E4M3FNType::get(ctx);
  case triton::ScaleDotElemType::E5M2:
    return Float8E5M2Type::get(ctx);
  case triton::ScaleDotElemType::E3M2:
    return Float6E3M2FNType::get(ctx);
  case triton::ScaleDotElemType::E2M3:
    return Float6E2M3FNType::get(ctx);
  case triton::ScaleDotElemType::E2M1:
    return Float4E2M1FNType::get(ctx);
  default:
    llvm_unreachable("unsupported ScaleDotElemType!");
  }
}

bool canCoalesceWriteIntoSharedMemory(RewriterBase &rewriter,
                                      const LinearLayout &srcToSharedLayout,
                                      unsigned threadsPerWarp) {
  auto contig = srcToSharedLayout.getNumConsecutiveInOut();

  StringAttr kLane = rewriter.getStringAttr("lane");
  for (int inLane : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kLane))) {
    auto basis = srcToSharedLayout.getBasis(kLane, inLane)[0];
    unsigned expected = contig * (1 << inLane);
    if (basis != expected) {
      LDBG("detected uncoalesced layout from blocked to shared in async copy "
           "for lane "
           << 1 + inLane << "; given " << basis << " but expected "
           << expected);
      return false;
    }
  }
  // Additionally we could swizzle based on the warp dimension so we need to
  // check that when all bases are divided by contig, none of the first
  // (log2(warpSize) + 1) bits are set to 1
  assert(llvm::isPowerOf2_32(threadsPerWarp));
  assert(llvm::isPowerOf2_32(contig));
  unsigned mask = (threadsPerWarp * contig) - 1;
  StringAttr kWarp = rewriter.getStringAttr("warp");
  for (int inWarp : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kWarp))) {
    auto basis = srcToSharedLayout.getBasis(kWarp, inWarp)[0];
    if ((basis & mask) != 0) {
      LDBG("detected uncoalesced layout from blocked to shared in async copy "
           "for warp "
           << inWarp);
      return false;
    }
  }

  return true;
}

bool doesSwizzleInsideWarp(RewriterBase &rewriter,
                           const LinearLayout &srcToSharedLayout,
                           unsigned threadsPerWarp) {
  auto contig = srcToSharedLayout.getNumConsecutiveInOut();
  // If all bases in lane dimension are below threadsPerWarp multiplied with the
  // contiguity we do not swizzle across warp boundaries.
  assert(llvm::isPowerOf2_32(threadsPerWarp));
  unsigned upperLimit = threadsPerWarp * contig;

  StringAttr kLane = rewriter.getStringAttr("lane");
  for (int inLane : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kLane))) {
    auto basis = srcToSharedLayout.getBasis(kLane, inLane)[0];
    if (basis >= upperLimit) {
      return false;
    }
  }
  return true;
}

bool isUsedByDotScaledOp(Operation *op) {
  const ForwardSliceOptions fwdOpt;
  SetVector<mlir::Operation *> forwardSliceSet;
  getForwardSlice(op, &forwardSliceSet, fwdOpt);

  return std::any_of(
      forwardSliceSet.begin(), forwardSliceSet.end(), [](auto *operation) {
        return isa<triton::DotScaledOp, triton::amdgpu::UpcastMXFPOp>(
            operation);
      });
}

bool isChainDotHead(tt::DotOpInterface dotOp) {
  auto isInSameRegion = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  ForwardSliceOptions fwdOpt;
  fwdOpt.filter = isInSameRegion;
  SetVector<mlir::Operation *> fwdSlices;
  getForwardSlice(dotOp, &fwdSlices, fwdOpt);
  for (Operation *op : fwdSlices) {
    if (auto dOp = dyn_cast<tt::DotOpInterface>(op)) {
      assert(dOp != dotOp);
      auto opA = dOp.getA().getDefiningOp();
      if (opA && fwdSlices.contains(opA)) {
        return true;
      }
    }
  }
  return false;
}

bool isChainDotTail(tt::DotOpInterface dotOp) {
  auto isInSameRegion = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = isInSameRegion;
  SetVector<Operation *> bwdSlices;
  Operation *opA = dotOp.getA().getDefiningOp();
  if (!opA)
    return false;
  (void)getBackwardSlice(opA, &bwdSlices, bwdOpt);
  if (llvm::find_if(bwdSlices, [](Operation *op) {
        return isa<tt::DotOpInterface>(op);
      }) != bwdSlices.end())
    return true;
  return false;
}
} // namespace mlir::LLVM::AMD
