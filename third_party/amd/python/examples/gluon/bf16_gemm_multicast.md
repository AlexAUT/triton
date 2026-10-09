# BF16 GEMM multicast reproducer on gfx1250

`bf16_gemm_multicast.py` contains the AITER-derived compute-bound Gluon GEMM and
its layout helpers. It requires a Triton build from this branch and a gfx1250 GPU.
It does not import AITER. Both configurations compute a 256×256 tile per CTA,
with BK=64, four warps, and four LDS buffers. The 4×4 cluster has a logical
1024×1024 tile; zero CTA bases in the operand layouts enable TDM multicast.

After installing this checkout (`python -m pip install -e . --no-build-isolation`),
run from the repository root. On the experiment machine:

```bash
source defaultEnv.sh && gpu-lock python third_party/amd/python/examples/gluon/bf16_gemm_multicast.py --ctas 1
source defaultEnv.sh && gpu-lock python third_party/amd/python/examples/gluon/bf16_gemm_multicast.py --ctas 16
```

The runner ignores source/IR overrides configured by `defaultEnv.sh` and uses
normal compilation caching. It prints correctness, median runtime, PFLOP/s,
register/spill counts, and static WMMA estimates if enabled by the environment.
Static estimates are not ATT hardware measurements. Defaults are M=N=K=8192 and
five timing rounds, with cache clearing before each measured dispatch. Use
`--m`, `--n`, `--k`, `--rounds`, or `--ctas 4` for additional comparisons. M and N
must be multiples of the logical cluster tile; K must be at least 384 and a
multiple of 16.

This source retains compiler-generated barriers and LDS waits. The separately
measured 3.106 PFLOP/s codeobject used an experimental, targeted LLVM IR fence
optimization that is not part of this branch. Runtime depends on the installed
compiler, clocks and shape; these commands measure the current normal compiler
path rather than guaranteeing historical numbers. The older matrix tutorial on
`aweinrau/gemmMatmulPerf` is a different kernel/compiler configuration.

Validation on upstream main `2a514c0f88` (2026-10-09), normal AMD LLVM pin
`6bc4aaf6` build 3: both commands passed correctness. Five-round medians were
0.43421 ms / 2.532 PFLOP/s single CTA and 0.40862 ms / 2.691 PFLOP/s multicast.
Both reported 956 VGPRs and zero spills. These measurements are machine-state
and compiler dependent, and are not the earlier optimized-codeobject result.
