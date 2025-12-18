/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* CUDA Common Definitions and Error Handling
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#ifndef SGM_CUDA_COMMON_CUH
#define SGM_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA Error Checking Macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, code=%d (%s)\n", \
                    __FILE__, __LINE__, error, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Check last CUDA error (for kernel launches)
#define CHECK_LAST_CUDA_ERROR() CHECK_CUDA(cudaGetLastError())

// CUDA Kernel Launch Checker
#define LAUNCH_KERNEL(kernel, grid, block, shared_mem, stream, ...) \
    do { \
        kernel<<<grid, block, shared_mem, stream>>>(__VA_ARGS__); \
        CHECK_LAST_CUDA_ERROR(); \
    } while(0)

// Common constants
// Census window: 9x7 means 9 rows (height) x 7 columns (width)
// This matches CPU implementation: r in [-4,4], c in [-3,3]
#define CENSUS_WIDTH 7   // horizontal (columns)
#define CENSUS_HEIGHT 9  // vertical (rows)
#define CENSUS_SIZE (CENSUS_WIDTH * CENSUS_HEIGHT - 1)  // 63 bits (exclude center)

#define INVALID_COST 255

// Block size configuration for RTX 4090
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

// Shared memory tile size for Census Transform (includes halo region)
#define TILE_WIDTH (BLOCK_WIDTH + CENSUS_WIDTH - 1)   // 32 + 8 = 40
#define TILE_HEIGHT (BLOCK_HEIGHT + CENSUS_HEIGHT - 1) // 32 + 6 = 38

#endif // SGM_CUDA_COMMON_CUH
