// RUN: triton-opt %s -split-input-file -verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @expanded_scale_fp4(
      %input: tensor<1x256xbf16, #unpacked>,
      %scale: tensor<1x128xi8, #scale>) {
    // expected-error @+1 {{expanded scales}}
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32}
        : tensor<1x256xbf16, #unpacked>, tensor<1x128xi8, #scale>
        -> tensor<1x128xi8, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale_compact = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scale_block_too_small_fp4(
      %input: tensor<1x256xbf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale_compact>) {
    // expected-error @+1 {{multiple of 8 fp4 values}}
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32}
        : tensor<1x256xbf16, #unpacked>, tensor<1x64xi8, #scale_compact>
        -> tensor<1x128xi8, #blocked>
    tt.return
  }
}

// -----

#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @expanded_scale_fp8(
      %input: tensor<1x256xbf16, #unpacked>,
      %scale: tensor<1x256xi8, #scale>) {
    // expected-error @+1 {{expanded scales}}
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32}
        : tensor<1x256xbf16, #unpacked>, tensor<1x256xi8, #scale>
        -> tensor<1x256xf8E4M3FN, #unpacked>
    tt.return
  }
}

// -----

#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scale_block_too_small_fp8(
      %input: tensor<1x256xbf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale>) {
    // expected-error @+1 {{multiple of 8 elements}}
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32}
        : tensor<1x256xbf16, #unpacked>, tensor<1x64xi8, #scale>
        -> tensor<1x256xf8E4M3FN, #unpacked>
    tt.return
  }
}
