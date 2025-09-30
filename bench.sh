#!/bin/bash

# batch_size=(32 16 8 4 2)
# seqlen=(512 1024 2048 4096 8192 16384)

# headim=(64 128)
headim=(128)
head_count_q=64
head_count_k=64
causal=(0)
layout="thd" # options: bshd bhsd thd

# BatchSize, SeqLen
# configs=(
#     "32 512"
#     "16 1024"
#     "8 2048"
#     "4 4096"
#     "2 8192"
#     "1 16384"
# )
configs=(
    # "32 512"
    # "16 1024"
    # "8 2048"
    # "4 4096"
    # "2 8192"
    "1 16384"
)

# Enable FAv3
export TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 AMDGCN_SCALARIZE_PACKED_FOPS=0
export TRITON_HIP_USE_PADDED_SHARED_LAYOUT=1

for c in "${causal[@]}"; do
    for h in "${headim[@]}"; do
        for batchSeq in "${configs[@]}"; do
            read -r b s <<< "$batchSeq"
            echo "Causal: $c, HeadDim: $h Batch size: $b SeqLen: $s"
            python3 fa/flash-attention.py -d "$h" -hq 64 -b "$b" -sq "$s" -causal "$c" -layout "$layout"
        done
    done
done
