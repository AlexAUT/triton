/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_DIALECT_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::triton::amd {
struct L2Cache : public SideEffects::Resource::Base<L2Cache> {
  StringRef getName() const final { return "<AMDGPU::L2Cache>"; }
};
} // namespace mlir::triton::amd

namespace mlir::triton::amdgpu {
/// Returns the number of dwords for a TDM tensor descriptor based on rank.
/// 2D tensors: group0 (4) + group1 (8) = 12 dwords
/// 3D-5D tensors: group0 (4) + group1 (8) + group2 (4) + group3 (4) = 20 dwords
inline int getTensorDescNumDwords(triton::TensorDescType type) {
  auto shape = type.getShape();
  return (shape.size() > 2) ? (4 + 8 + 4 + 4) : (4 + 8);
}
} // namespace mlir::triton::amdgpu

// clang-format off
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h.inc"
#include "amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUEnums.h.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.h.inc"

#include "amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOpInterfaces.h.inc"
#define GET_OP_CLASSES
#include "amd/include/Dialect/TritonAMDGPU/IR/Ops.h.inc"

namespace mlir::triton::amdgpu {

struct Fp4Pk8ScaleOpSel {
  int32_t scaleBase;
  int32_t scaleSel;
  bool duplicateScalePairs;
};

/// Return the scale-register packing and scale_sel value for each packed FP4
/// conversion group when the scale layout can be consumed directly by
/// v_cvt_scale_pk8's four scale selectors.
std::optional<SmallVector<Fp4Pk8ScaleOpSel>>
computeFp4Pk8ScaleOpSel(ScaledUpcastFp4Op op);

} // namespace mlir::triton::amdgpu

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_DIALECT_H_
