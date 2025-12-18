/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Validation Utilities: Compare CPU vs GPU Results
*/

#ifndef SGM_CUDA_VALIDATION_H
#define SGM_CUDA_VALIDATION_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

/**
 * @brief Validate Census Transform results
 * 
 * Compares GPU Census output against CPU reference implementation.
 * Reports mismatches and statistics.
 * 
 * @param cpu_census CPU-computed Census values
 * @param gpu_census GPU-computed Census values (copied from device)
 * @param width Image width
 * @param height Image height
 * @param verbose Print detailed mismatch information
 * @return true if results match (or within tolerance)
 */
inline bool validate_census(
    const uint64_t* cpu_census,
    const uint64_t* gpu_census,
    int width,
    int height,
    bool verbose = false
) {
    int mismatches = 0;
    int total_pixels = width * height;
    
    printf("\n========== Census Validation ==========\n");
    printf("Resolution: %d x %d (%d pixels)\n", width, height, total_pixels);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            uint64_t cpu_val = cpu_census[idx];
            uint64_t gpu_val = gpu_census[idx];
            
            if (cpu_val != gpu_val) {
                mismatches++;
                if (verbose && mismatches <= 10) {
                    printf("  Mismatch at (%d, %d): CPU=0x%016llx, GPU=0x%016llx\n",
                           x, y, (unsigned long long)cpu_val, (unsigned long long)gpu_val);
                }
            }
        }
    }
    
    float error_rate = (float)mismatches / total_pixels * 100.0f;
    printf("Mismatches: %d / %d (%.4f%%)\n", mismatches, total_pixels, error_rate);
    
    bool passed = (error_rate < 0.01f); // Allow < 0.01% error
    printf("Result: %s\n", passed ? "PASSED" : "FAILED");
    printf("=======================================\n\n");
    
    return passed;
}

/**
 * @brief Validate Cost Volume results
 * 
 * Compares GPU cost volume against CPU reference.
 * Uses statistical sampling for large volumes.
 * 
 * @param cpu_cost CPU-computed cost volume
 * @param gpu_cost GPU-computed cost volume (copied from device)
 * @param width Image width
 * @param height Image height
 * @param max_disparity Maximum disparity
 * @param tolerance Maximum allowed difference per pixel
 * @param verbose Print detailed mismatch information
 * @return true if results match within tolerance
 */
inline bool validate_cost_volume(
    const unsigned char* cpu_cost,
    const unsigned char* gpu_cost,
    int width,
    int height,
    int max_disparity,
    int tolerance = 1,
    bool verbose = false
) {
    int mismatches = 0;
    int total_elements = width * height * max_disparity;
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    
    printf("\n========== Cost Volume Validation ==========\n");
    printf("Dimensions: %d x %d x %d (%d elements)\n",
           width, height, max_disparity, total_elements);
    printf("Tolerance: %d\n", tolerance);
    
    // Full comparison (can be slow for large volumes)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int d = 0; d < max_disparity; d++) {
                int idx = y * (width * max_disparity) + x * max_disparity + d;
                int cpu_val = cpu_cost[idx];
                int gpu_val = gpu_cost[idx];
                int diff = abs(cpu_val - gpu_val);
                
                avg_diff += diff;
                if (diff > max_diff) max_diff = diff;
                
                if (diff > tolerance) {
                    mismatches++;
                    if (verbose && mismatches <= 10) {
                        printf("  Mismatch at (%d,%d,d=%d): CPU=%d, GPU=%d, diff=%d\n",
                               x, y, d, cpu_val, gpu_val, diff);
                    }
                }
            }
        }
    }
    
    avg_diff /= total_elements;
    float error_rate = (float)mismatches / total_elements * 100.0f;
    
    printf("Mismatches (diff > %d): %d / %d (%.4f%%)\n",
           tolerance, mismatches, total_elements, error_rate);
    printf("Max difference: %.2f\n", max_diff);
    printf("Avg difference: %.4f\n", avg_diff);
    
    bool passed = (error_rate < 0.1f); // Allow < 0.1% error
    printf("Result: %s\n", passed ? "PASSED" : "FAILED");
    printf("============================================\n\n");
    
    return passed;
}

/**
 * @brief Quick sample validation for cost volume
 * 
 * Validates a random sample of cost volume elements (faster for large volumes)
 */
inline bool validate_cost_volume_sampled(
    const unsigned char* cpu_cost,
    const unsigned char* gpu_cost,
    int width,
    int height,
    int max_disparity,
    int num_samples = 10000,
    int tolerance = 1
) {
    int mismatches = 0;
    int total_elements = width * height * max_disparity;
    
    printf("\n========== Cost Volume Validation (Sampled) ==========\n");
    printf("Total elements: %d, Samples: %d\n", total_elements, num_samples);
    
    srand(42); // Fixed seed for reproducibility
    
    for (int i = 0; i < num_samples; i++) {
        int idx = rand() % total_elements;
        int diff = abs((int)cpu_cost[idx] - (int)gpu_cost[idx]);
        
        if (diff > tolerance) {
            mismatches++;
        }
    }
    
    float error_rate = (float)mismatches / num_samples * 100.0f;
    printf("Mismatches: %d / %d (%.4f%%)\n", mismatches, num_samples, error_rate);
    
    bool passed = (error_rate < 0.1f);
    printf("Result: %s\n", passed ? "PASSED" : "FAILED");
    printf("======================================================\n\n");
    
    return passed;
}

#endif // SGM_CUDA_VALIDATION_H
