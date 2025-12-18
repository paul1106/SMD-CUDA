/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Test Program: Phase 2 Cost Aggregation Validation (Horizontal + Vertical)
* Compares CUDA implementation against CPU reference (with fixed P2)
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cstring>
#include <algorithm>

// CPU reference implementation
#include "../SemiGlobalMatching/sgm_util.h"
#include "../SemiGlobalMatching/sgm_types.h"

// CUDA implementation
#include "include/sgm_cuda_phase1.cuh"
#include "include/sgm_cuda_phase2.cuh"
#include "include/sgm_cuda_validation.h"

using namespace std::chrono;

void print_usage(const char* prog_name) {
    printf("Usage: %s <left_image> <right_image> <max_disparity>\n", prog_name);
    printf("Example: %s Data/cone/im2.png Data/cone/im6.png 64\n", prog_name);
}

/**
 * CPU Reference: Horizontal Aggregation with FIXED P2
 * (Modified from sgm_util to use fixed P2 instead of adaptive)
 * 
 * NOTE: This writes to cost_aggr, not accumulates
 */
void cpu_horizontal_aggregation_fixed_p2(
    const uint8* cost_init,
    uint16* cost_aggr, // Output (not accumulated, written)
    int width,
    int height,
    int disp_range,
    int p1,
    int p2,
    bool is_forward
) {
    const int direction = is_forward ? 1 : -1;
    
    for (int y = 0; y < height; y++) {
        auto cost_init_row = is_forward ? 
            (cost_init + y * width * disp_range) : 
            (cost_init + y * width * disp_range + (width - 1) * disp_range);
            
        auto cost_aggr_row = is_forward ? 
            (cost_aggr + y * width * disp_range) : 
            (cost_aggr + y * width * disp_range + (width - 1) * disp_range);
        
        // Store previous path costs (with padding for d-1 and d+1 access)
        // Note: Use 255 (UINT8_MAX) for boundary padding to match original CPU implementation
        std::vector<uint16> cost_last_path(disp_range + 2, 255);
        
        // Initialize first pixel - just copy raw cost
        for (int d = 0; d < disp_range; d++) {
            cost_last_path[d + 1] = cost_init_row[d];
            cost_aggr_row[d] = cost_init_row[d]; // Write (not accumulate)
        }
        
        cost_init_row += direction * disp_range;
        cost_aggr_row += direction * disp_range;
        
        // Find min of previous path (skip padding elements at index 0 and disp_range+1)
        uint16 mincost_last_path = 255;
        for (int d = 1; d <= disp_range; d++) {
            mincost_last_path = std::min(mincost_last_path, cost_last_path[d]);
        }
        
        // Aggregate remaining pixels in row
        for (int x = 1; x < width; x++) {
            uint16 min_cost = UINT16_MAX;
            
            for (int d = 0; d < disp_range; d++) {
                // SGM formula with fixed P2
                const uint16 cost = cost_init_row[d];
                const uint16 l1 = cost_last_path[d + 1];
                const uint16 l2 = cost_last_path[d] + p1;
                const uint16 l3 = cost_last_path[d + 2] + p1;
                const uint16 l4 = mincost_last_path + p2;
                
                uint16 cost_s = cost + std::min({l1, l2, l3, l4}) - mincost_last_path;
                
                cost_aggr_row[d] = cost_s; // Write (not accumulate)
                min_cost = std::min(min_cost, cost_s);
            }
            
            // Update cost_last_path after processing all disparities
            mincost_last_path = min_cost;
            std::memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint16));
            
            cost_init_row += direction * disp_range;
            cost_aggr_row += direction * disp_range;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return -1;
    }

    // ===== Parse Arguments =====
    std::string path_left = argv[1];
    std::string path_right = argv[2];
    int max_disparity = atoi(argv[3]);

    printf("==============================================\n");
    printf("SGM CUDA Phase 2 Validation Test\n");
    printf("(Horizontal Aggregation Only)\n");
    printf("==============================================\n");
    printf("Left Image:  %s\n", path_left.c_str());
    printf("Right Image: %s\n", path_right.c_str());
    printf("Max Disparity: %d\n", max_disparity);
    printf("Parameters: P1=%d, P2=%d (fixed)\n", SGM_P1, SGM_P2);
    printf("==============================================\n\n");

    // ===== Load Images =====
    cv::Mat img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

    if (img_left.empty() || img_right.empty()) {
        fprintf(stderr, "Error: Cannot load images\n");
        return -1;
    }

    const int width = img_left.cols;
    const int height = img_left.rows;
    const size_t img_size = width * height;
    const size_t cost_volume_size = img_size * max_disparity;

    printf("Image Size: %d x %d\n\n", width, height);

    // ===== Prepare Buffers =====
    auto* h_imL = new uint8[img_size];
    auto* h_imR = new uint8[img_size];
    auto* h_cost_volume = new uint8[cost_volume_size];
    auto* h_aggr_cpu = new uint16[cost_volume_size]();
    auto* h_aggr_gpu = new uint16[cost_volume_size]();

    // Copy image data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_imL[y * width + x] = img_left.at<uint8>(y, x);
            h_imR[y * width + x] = img_right.at<uint8>(y, x);
        }
    }

    // ===== Phase 1: Generate Cost Volume (GPU) =====
    printf("Running Phase 1 (GPU) to generate cost volume...\n");
    
    unsigned char *d_imL, *d_imR, *d_cost_volume;
    uint64_t *d_censusL, *d_censusR;
    
    CHECK_CUDA(cudaMalloc(&d_imL, img_size));
    CHECK_CUDA(cudaMalloc(&d_imR, img_size));
    CHECK_CUDA(cudaMalloc(&d_censusL, img_size * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_censusR, img_size * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_cost_volume, cost_volume_size));
    
    CHECK_CUDA(cudaMemcpy(d_imL, h_imL, img_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_imR, h_imR, img_size, cudaMemcpyHostToDevice));
    
    launch_phase1(d_imL, d_imR, d_censusL, d_censusR, d_cost_volume,
                  width, height, max_disparity);
    
    CHECK_CUDA(cudaMemcpy(h_cost_volume, d_cost_volume, cost_volume_size, cudaMemcpyDeviceToHost));
    printf("Phase 1 completed.\n\n");

    // ===== CPU Reference: Horizontal Aggregation =====
    printf("Running CPU Horizontal Aggregation (Fixed P2)...\n");
    auto cpu_start = steady_clock::now();
    
    // CPU needs separate buffers for each path, then accumulates
    auto* h_aggr_lr = new uint16[cost_volume_size]();
    auto* h_aggr_rl = new uint16[cost_volume_size]();
    
    cpu_horizontal_aggregation_fixed_p2(h_cost_volume, h_aggr_lr, 
        width, height, max_disparity, SGM_P1, SGM_P2, true);  // Left-to-right
    cpu_horizontal_aggregation_fixed_p2(h_cost_volume, h_aggr_rl, 
        width, height, max_disparity, SGM_P1, SGM_P2, false); // Right-to-left
    
    // Debug: Print single direction results
    printf("\nDebug: CPU paths (first pixel at y=0, x=0):\n");
    printf("d\tRaw\tL->R\tR->L\tSum\n");
    for (int d = 0; d < std::min(10, max_disparity); d++) {
        printf("%d\t%d\t%d\t%d\t%d\n", d, h_cost_volume[d], 
               h_aggr_lr[d], h_aggr_rl[d], h_aggr_lr[d] + h_aggr_rl[d]);
    }
    
    printf("\nDebug: CPU paths (first pixel at y=0, x=1):\n");
    printf("d\tRaw\tL->R\tR->L\tSum\n");
    for (int d = 0; d < std::min(10, max_disparity); d++) {
        int idx = max_disparity + d;  // x=1
        printf("%d\t%d\t%d\t%d\t%d\n", d, h_cost_volume[idx], 
               h_aggr_lr[idx], h_aggr_rl[idx], h_aggr_lr[idx] + h_aggr_rl[idx]);
    }
    printf("\n");
    
    // Accumulate both paths
    for (size_t i = 0; i < cost_volume_size; i++) {
        h_aggr_cpu[i] = h_aggr_lr[i] + h_aggr_rl[i];
    }
    
    delete[] h_aggr_lr;
    delete[] h_aggr_rl;
    
    auto cpu_end = steady_clock::now();
    double cpu_time = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    printf("  CPU Time: %.2f ms\n\n", cpu_time);

    // ===== GPU: Phase 2 Horizontal Aggregation =====
    printf("Running GPU Phase 2 Horizontal Aggregation...\n");
    
    uint16_t* d_total_cost;
    CHECK_CUDA(cudaMalloc(&d_total_cost, cost_volume_size * sizeof(uint16_t)));
    CHECK_CUDA(cudaMemset(d_total_cost, 0, cost_volume_size * sizeof(uint16_t)));
    
    auto gpu_start = steady_clock::now();
    
    launch_phase2_horizontal(d_cost_volume, d_total_cost, width, height, max_disparity);
    
    auto gpu_end = steady_clock::now();
    double gpu_time = duration_cast<milliseconds>(gpu_end - gpu_start).count();
    printf("  GPU Time: %.2f ms\n", gpu_time);
    printf("  Speedup: %.2fx\n\n", cpu_time / gpu_time);

    // ===== Copy Results Back =====
    CHECK_CUDA(cudaMemcpy(h_aggr_gpu, d_total_cost, cost_volume_size * sizeof(uint16_t), 
                          cudaMemcpyDeviceToHost));

    // ===== Validation =====
    printf("========== VALIDATION ==========\n");
    
    // Debug: Print first few values
    printf("\nDebug: First pixel (0,0) aggregated costs:\n");
    printf("d\tRaw\tCPU\tGPU\tDiff\n");
    for (int d = 0; d < std::min(10, max_disparity); d++) {
        int idx = d; // y=0, x=0
        printf("%d\t%d\t%d\t%d\t%d\n", d, h_cost_volume[idx], 
               h_aggr_cpu[idx], h_aggr_gpu[idx], 
               abs((int)h_aggr_cpu[idx] - (int)h_aggr_gpu[idx]));
    }
    printf("\n");
    
    int mismatches = 0;
    int total = cost_volume_size;
    uint64_t total_diff = 0;
    int max_diff = 0;
    
    for (int i = 0; i < total; i++) {
        int diff = abs((int)h_aggr_cpu[i] - (int)h_aggr_gpu[i]);
        total_diff += diff;
        max_diff = std::max(max_diff, diff);
        
        if (diff > 1) { // Allow tolerance of 1 due to rounding
            mismatches++;
            if (mismatches <= 5) {
                int y = i / (width * max_disparity);
                int x = (i % (width * max_disparity)) / max_disparity;
                int d = i % max_disparity;
                printf("  Mismatch at (%d,%d,d=%d): CPU=%d, GPU=%d, diff=%d\n",
                       x, y, d, h_aggr_cpu[i], h_aggr_gpu[i], diff);
            }
        }
    }
    
    double avg_diff = (double)total_diff / total;
    double error_rate = (double)mismatches / total * 100.0;
    
    printf("\nMismatches (diff > 1): %d / %d (%.4f%%)\n", mismatches, total, error_rate);
    printf("Max difference: %d\n", max_diff);
    printf("Avg difference: %.4f\n", avg_diff);
    
    bool passed = (error_rate < 0.1);
    printf("Result: %s\n", passed ? "PASSED" : "FAILED");
    printf("================================\n\n");

    // ===== Compute and Visualize Disparity Maps =====
    printf("Computing disparity maps...\n");
    
    auto* disp_cpu = new float[width * height];
    auto* disp_gpu = new float[width * height];
    
    // Compute disparity from aggregated costs (WTA - Winner Takes All)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx_pixel = y * width + x;
            
            // CPU disparity
            uint16_t min_cost_cpu = UINT16_MAX;
            int best_d_cpu = 0;
            for (int d = 0; d < max_disparity; d++) {
                int idx = y * width * max_disparity + x * max_disparity + d;
                if (h_aggr_cpu[idx] < min_cost_cpu) {
                    min_cost_cpu = h_aggr_cpu[idx];
                    best_d_cpu = d;
                }
            }
            disp_cpu[idx_pixel] = best_d_cpu;
            
            // GPU disparity
            uint16_t min_cost_gpu = UINT16_MAX;
            int best_d_gpu = 0;
            for (int d = 0; d < max_disparity; d++) {
                int idx = y * width * max_disparity + x * max_disparity + d;
                if (h_aggr_gpu[idx] < min_cost_gpu) {
                    min_cost_gpu = h_aggr_gpu[idx];
                    best_d_gpu = d;
                }
            }
            disp_gpu[idx_pixel] = best_d_gpu;
        }
    }
    
    // Normalize and visualize
    cv::Mat disp_cpu_mat(height, width, CV_32F, disp_cpu);
    cv::Mat disp_gpu_mat(height, width, CV_32F, disp_gpu);
    
    cv::Mat disp_cpu_vis, disp_gpu_vis;
    disp_cpu_mat.convertTo(disp_cpu_vis, CV_8U, 255.0 / max_disparity);
    disp_gpu_mat.convertTo(disp_gpu_vis, CV_8U, 255.0 / max_disparity);
    
    // Apply colormap for better visualization
    cv::Mat disp_cpu_color, disp_gpu_color;
    cv::applyColorMap(disp_cpu_vis, disp_cpu_color, cv::COLORMAP_JET);
    cv::applyColorMap(disp_gpu_vis, disp_gpu_color, cv::COLORMAP_JET);
    
    // Create side-by-side comparison
    cv::Mat comparison(height, width * 2, CV_8UC3);
    disp_cpu_color.copyTo(comparison(cv::Rect(0, 0, width, height)));
    disp_gpu_color.copyTo(comparison(cv::Rect(width, 0, width, height)));
    
    // Add labels
    cv::putText(comparison, "CPU Reference", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "GPU CUDA", cv::Point(width + 10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    
    // Save output
    cv::imwrite("disparity_comparison.png", comparison);
    printf("Disparity comparison saved to: disparity_comparison.png\n");
    printf("  Left: CPU Reference | Right: GPU CUDA\n\n");
    
    delete[] disp_cpu;
    delete[] disp_gpu;

    // ===== Summary =====
    printf("==============================================\n");
    printf("SUMMARY\n");
    printf("==============================================\n");
    printf("Horizontal Aggregation: %s\n", passed ? "PASSED" : "FAILED");
    printf("CPU Time: %.2f ms\n", cpu_time);
    printf("GPU Time: %.2f ms\n", gpu_time);
    printf("Speedup:  %.2fx\n", cpu_time / gpu_time);
    printf("Output: disparity_comparison.png\n");
    printf("==============================================\n");

    // ===== Cleanup =====
    delete[] h_imL;
    delete[] h_imR;
    delete[] h_cost_volume;
    delete[] h_aggr_cpu;
    delete[] h_aggr_gpu;

    cudaFree(d_imL);
    cudaFree(d_imR);
    cudaFree(d_censusL);
    cudaFree(d_censusR);
    cudaFree(d_cost_volume);
    cudaFree(d_total_cost);

    return passed ? 0 : 1;
}
