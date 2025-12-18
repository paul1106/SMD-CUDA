/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Implementation: Census Transform Kernel with Shared Memory Optimization
* Target: NVIDIA RTX 4090 (Compute Capability 8.9)
*/

#include "../include/sgm_cuda_phase1.cuh"

/**
 * Census Transform Kernel with Tile-based Shared Memory
 * 
 * Strategy:
 * 1. Each block processes a 32x32 tile of pixels
 * 2. Loads 40x38 tile (includes halo region for 9x7 window) into shared memory
 * 3. Each thread computes Census from shared memory (reduces global memory reads ~40x)
 * 4. Writes result to global memory
 * 
 * Memory Access Pattern:
 * - Global reads: ~1.5 per thread (with halo sharing)
 * - Shared reads: 63 per thread
 * - Global writes: 1 per thread
 */
__global__ void census_transform_kernel(
    const unsigned char* __restrict__ image,
    uint64_t* __restrict__ census,
    int width,
    int height
) {
    // Shared memory tile: includes halo region for 9x7 window
    __shared__ unsigned char tile[TILE_HEIGHT][TILE_WIDTH];
    
    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_WIDTH + tx;
    const int y = blockIdx.y * BLOCK_HEIGHT + ty;
    
    // Starting position of the tile in global memory
    const int tile_start_x = blockIdx.x * BLOCK_WIDTH - CENSUS_WIDTH / 2;
    const int tile_start_y = blockIdx.y * BLOCK_HEIGHT - CENSUS_HEIGHT / 2;
    
    // ===== Phase 1: Collaborative Loading of Tile + Halo =====
    // Each thread may load multiple pixels to cover the entire tile
    const int tile_pixels = TILE_WIDTH * TILE_HEIGHT;
    const int threads_per_block = BLOCK_WIDTH * BLOCK_HEIGHT;
    const int thread_id = ty * BLOCK_WIDTH + tx;
    
    for (int i = thread_id; i < tile_pixels; i += threads_per_block) {
        const int tile_y = i / TILE_WIDTH;
        const int tile_x = i % TILE_WIDTH;
        
        const int global_x = tile_start_x + tile_x;
        const int global_y = tile_start_y + tile_y;
        
        // Boundary check: pad with 0 for out-of-bounds
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            tile[tile_y][tile_x] = image[global_y * width + global_x];
        } else {
            tile[tile_y][tile_x] = 0;
        }
    }
    
    // Synchronize to ensure tile is fully loaded
    __syncthreads();
    
    // ===== Phase 2: Census Computation from Shared Memory =====
    if (x >= width || y >= height) return;
    
    // Center position in tile (accounting for halo offset)
    const int center_tile_x = tx + CENSUS_WIDTH / 2;
    const int center_tile_y = ty + CENSUS_HEIGHT / 2;
    
    // Check if we're too close to image boundaries
    const bool valid = (x >= CENSUS_WIDTH / 2 && x < width - CENSUS_WIDTH / 2 &&
                        y >= CENSUS_HEIGHT / 2 && y < height - CENSUS_HEIGHT / 2);
    
    if (!valid) {
        census[y * width + x] = 0;
        return;
    }
    
    const unsigned char center = tile[center_tile_y][center_tile_x];
    uint64_t census_value = 0;
    
    // Bit-pack comparison results: (neighbor < center) ? 1 : 0
    // Match CPU implementation: shift left first, then add bit
    // Window: r in [-4, 4] (9 rows), c in [-3, 3] (7 columns)
    
    #pragma unroll
    for (int r = -CENSUS_HEIGHT / 2; r <= CENSUS_HEIGHT / 2; r++) {
        #pragma unroll
        for (int c = -CENSUS_WIDTH / 2; c <= CENSUS_WIDTH / 2; c++) {
            // Shift left for every pixel (including center)
            census_value <<= 1;
            
            const unsigned char neighbor = tile[center_tile_y + r][center_tile_x + c];
            
            // Add 1 if neighbor < center
            if (neighbor < center) {
                census_value += 1;
            }
        }
    }
    
    // Write result to global memory
    census[y * width + x] = census_value;
}
