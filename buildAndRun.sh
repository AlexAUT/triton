#! /usr/bin/bash

set -e

rm -f out.amdgcn out.mlir

cmake --build python/build/cmake.linux-x86_64-cpython-3.10 --parallel

export TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 MLIR_DUMP_PATH=out.mlir AMDGCN_DUMP_PATH=out.amdgcn
export TRITON_LLVM_DEBUG_ONLY="tritonamdgpu-coalesce-async-copy"
python python/tutorials/03-matrix-multiplication.py
