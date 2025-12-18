/* -*-c++-*- SemiGlobalMatching CUDA - Copyright (C) 2025.
* Test Program: Phase 2 Full Aggregation (4 Paths: H + V)
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
#include "include/sgm_cuda_phase4.cuh"
#include "include/sgm_cuda_validation.h"

using namespace std::chrono;

void print_usage(const char* prog_name) {
    printf("Usage: %s <left_image> <right_image> <max_disparity>\n", prog_name);
    printf("Example: %s Data/cone/im2.png Data/cone/im6.png 64\n", prog_name);
}

/**
 * CPU Reference: Horizontal/Vertical Aggregation with FIXED P2
 */
void cpu_aggregation_fixed_p2(
    const uint8* cost_init,
    uint16* cost_aggr,
    int width,
    int height,
    int disp_range,
    int p1,
    int p2,
    bool is_horizontal,  // true for horizontal, false for vertical
    bool is_forward
) {
    const int direction = is_forward ? 1 : -1;
    
    if (is_horizontal) {
        // Horizontal aggregation (same as before)
        for (int y = 0; y < height; y++) {
            auto cost_init_row = is_forward ? 
                (cost_init + y * width * disp_range) : 
                (cost_init + y * width * disp_range + (width - 1) * disp_range);
                
            auto cost_aggr_row = is_forward ? 
                (cost_aggr + y * width * disp_range) : 
                (cost_aggr + y * width * disp_range + (width - 1) * disp_range);
            
            std::vector<uint16> cost_last_path(disp_range + 2, 255);
            
            for (int d = 0; d < disp_range; d++) {
                cost_last_path[d + 1] = cost_init_row[d];
                cost_aggr_row[d] = cost_init_row[d];
            }
            
            cost_init_row += direction * disp_range;
            cost_aggr_row += direction * disp_range;
            
            uint16 mincost_last_path = 255;
            for (int d = 1; d <= disp_range; d++) {
                mincost_last_path = std::min(mincost_last_path, cost_last_path[d]);
            }
            
            for (int x = 1; x < width; x++) {
                uint16 min_cost = UINT16_MAX;
                
                for (int d = 0; d < disp_range; d++) {
                    const uint16 cost = cost_init_row[d];
                    const uint16 l1 = cost_last_path[d + 1];
                    const uint16 l2 = cost_last_path[d] + p1;
                    const uint16 l3 = cost_last_path[d + 2] + p1;
                    const uint16 l4 = mincost_last_path + p2;
                    
                    uint16 cost_s = cost + std::min({l1, l2, l3, l4}) - mincost_last_path;
                    
                    cost_aggr_row[d] = cost_s;
                    min_cost = std::min(min_cost, cost_s);
                }
                
                mincost_last_path = min_cost;
                std::memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint16));
                
                cost_init_row += direction * disp_range;
                cost_aggr_row += direction * disp_range;
            }
        }
    } else {
        // Vertical aggregation
        for (int x = 0; x < width; x++) {
            auto cost_init_col = is_forward ?
                (cost_init + x * disp_range) :
                (cost_init + (height - 1) * width * disp_range + x * disp_range);
                
            auto cost_aggr_col = is_forward ?
                (cost_aggr + x * disp_range) :
                (cost_aggr + (height - 1) * width * disp_range + x * disp_range);
            
            std::vector<uint16> cost_last_path(disp_range + 2, 255);
            
            for (int d = 0; d < disp_range; d++) {
                cost_last_path[d + 1] = cost_init_col[d];
                cost_aggr_col[d] = cost_init_col[d];
            }
            
            cost_init_col += direction * width * disp_range;
            cost_aggr_col += direction * width * disp_range;
            
            uint16 mincost_last_path = 255;
            for (int d = 1; d <= disp_range; d++) {
                mincost_last_path = std::min(mincost_last_path, cost_last_path[d]);
            }
            
            for (int y = 1; y < height; y++) {
                uint16 min_cost = UINT16_MAX;
                
                for (int d = 0; d < disp_range; d++) {
                    const uint16 cost = cost_init_col[d];
                    const uint16 l1 = cost_last_path[d + 1];
                    const uint16 l2 = cost_last_path[d] + p1;
                    const uint16 l3 = cost_last_path[d + 2] + p1;
                    const uint16 l4 = mincost_last_path + p2;
                    
                    uint16 cost_s = cost + std::min({l1, l2, l3, l4}) - mincost_last_path;
                    
                    cost_aggr_col[d] = cost_s;
                    min_cost = std::min(min_cost, cost_s);
                }
                
                mincost_last_path = min_cost;
                std::memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint16));
                
                cost_init_col += direction * width * disp_range;
                cost_aggr_col += direction * width * disp_range;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return -1;
    }

    std::string path_left = argv[1];
    std::string path_right = argv[2];
    int max_disparity = atoi(argv[3]);
    
    // Extract dataset name from path for output filename
    std::string dataset_name = "output";
    size_t last_slash = path_left.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        size_t second_last_slash = path_left.find_last_of("/\\", last_slash - 1);
        if (second_last_slash != std::string::npos) {
            dataset_name = path_left.substr(second_last_slash + 1, last_slash - second_last_slash - 1);
        }
    }

    printf("==============================================\n");
    printf("SGM CUDA Phase 2 Full Aggregation Test\n");
    printf("(4 Paths: Horizontal + Vertical)\n");
    printf("==============================================\n");
    printf("Left Image:  %s\n", path_left.c_str());
    printf("Right Image: %s\n", path_right.c_str());
    printf("Max Disparity: %d\n", max_disparity);
    printf("Parameters: P1=%d, P2=%d (fixed)\n", SGM_P1, SGM_P2);
    printf("==============================================\n\n");

    cv::Mat img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

    if (img_left.empty() || img_right.empty()) {
        fprintf(stderr, "Error: Cannot load images\n");
        return -1;
    }

    const int width = img_left.cols;
    const int height = img_left.rows;
    const size_t img_size = width * height;
    const size_t cost_volume_size = width * height * max_disparity;

    printf("Image Size: %d x %d\n\n", width, height);

    auto* h_imL = new uint8[img_size];
    auto* h_imR = new uint8[img_size];
    auto* h_cost_volume = new uint8[cost_volume_size];
    auto* h_aggr_cpu = new uint16[cost_volume_size]();
    auto* h_aggr_gpu = new uint16[cost_volume_size]();

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

    // ===== CPU Reference: 4-Path Aggregation =====
    printf("Running CPU 4-Path Aggregation (Fixed P2)...\n");
    auto cpu_start = steady_clock::now();
    
    auto* h_aggr_lr = new uint16[cost_volume_size]();
    auto* h_aggr_rl = new uint16[cost_volume_size]();
    auto* h_aggr_ud = new uint16[cost_volume_size]();
    auto* h_aggr_du = new uint16[cost_volume_size]();
    
    // Horizontal paths
    cpu_aggregation_fixed_p2(h_cost_volume, h_aggr_lr, 
        width, height, max_disparity, SGM_P1, SGM_P2, true, true);
    cpu_aggregation_fixed_p2(h_cost_volume, h_aggr_rl, 
        width, height, max_disparity, SGM_P1, SGM_P2, true, false);
    
    // Vertical paths
    cpu_aggregation_fixed_p2(h_cost_volume, h_aggr_ud, 
        width, height, max_disparity, SGM_P1, SGM_P2, false, true);
    cpu_aggregation_fixed_p2(h_cost_volume, h_aggr_du, 
        width, height, max_disparity, SGM_P1, SGM_P2, false, false);
    
    // Accumulate all 4 paths
    for (size_t i = 0; i < cost_volume_size; i++) {
        h_aggr_cpu[i] = h_aggr_lr[i] + h_aggr_rl[i] + h_aggr_ud[i] + h_aggr_du[i];
    }
    
    delete[] h_aggr_lr;
    delete[] h_aggr_rl;
    delete[] h_aggr_ud;
    delete[] h_aggr_du;
    
    auto cpu_end = steady_clock::now();
    double cpu_time = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    printf("  CPU Time: %.2f ms\n\n", cpu_time);

    // ===== GPU: Phase 2 Full Aggregation (4 Paths) =====
    printf("Running GPU Phase 2 Full Aggregation (4 Paths)...\n");
    
    uint16_t* d_total_cost;
    CHECK_CUDA(cudaMalloc(&d_total_cost, cost_volume_size * sizeof(uint16_t)));
    CHECK_CUDA(cudaMemset(d_total_cost, 0, cost_volume_size * sizeof(uint16_t)));
    
    auto gpu_start = steady_clock::now();
    
    // Horizontal paths (L->R, R->L)
    launch_phase2_horizontal(d_cost_volume, d_total_cost, width, height, max_disparity);
    
    // Vertical paths (U->D, D->U) via transpose strategy
    launch_vertical_aggregation(d_cost_volume, d_total_cost, width, height, max_disparity);
    
    auto gpu_end = steady_clock::now();
    double gpu_time = duration_cast<milliseconds>(gpu_end - gpu_start).count();
    printf("  GPU Time: %.2f ms\n", gpu_time);
    printf("  Speedup: %.2fx\n\n", cpu_time / gpu_time);

    // ===== Copy Results Back =====
    CHECK_CUDA(cudaMemcpy(h_aggr_gpu, d_total_cost, cost_volume_size * sizeof(uint16_t), 
                          cudaMemcpyDeviceToHost));

    // ===== Validation =====
    printf("========== VALIDATION ==========\n");
    
    int mismatches = 0;
    int total = cost_volume_size;
    uint64_t total_diff = 0;
    int max_diff = 0;
    
    for (int i = 0; i < total; i++) {
        int diff = abs((int)h_aggr_cpu[i] - (int)h_aggr_gpu[i]);
        total_diff += diff;
        max_diff = std::max(max_diff, diff);
        
        if (diff > 1) {
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

    // ===== Phase 4: Winner-Takes-All (Disparity Computation) =====
    printf("\n========== PHASE 4: DISPARITY COMPUTATION ==========\n");
    
    // CPU WTA (for validation)
    printf("Computing CPU disparity map...\n");
    auto cpu_wta_start = steady_clock::now();
    
    auto* disp_cpu = new float[width * height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx_pixel = y * width + x;
            
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
        }
    }
    
    auto cpu_wta_end = steady_clock::now();
    double cpu_wta_time = duration_cast<milliseconds>(cpu_wta_end - cpu_wta_start).count();
    printf("  CPU WTA Time: %.2f ms\n\n", cpu_wta_time);
    
    // GPU WTA
    printf("Computing GPU disparity map...\n");
    
    unsigned char* d_disparity_map;
    CHECK_CUDA(cudaMalloc(&d_disparity_map, width * height));
    
    auto gpu_wta_start = steady_clock::now();
    
    launch_wta(d_total_cost, d_disparity_map, width, height, max_disparity);
    
    auto gpu_wta_end = steady_clock::now();
    double gpu_wta_time = duration_cast<milliseconds>(gpu_wta_end - gpu_wta_start).count();
    printf("  GPU WTA Time: %.2f ms\n", gpu_wta_time);
    printf("  WTA Speedup: %.2fx\n\n", cpu_wta_time / gpu_wta_time);
    
    // Copy GPU disparity map back (scaled 0-255)
    auto* disp_gpu_scaled = new unsigned char[width * height];
    CHECK_CUDA(cudaMemcpy(disp_gpu_scaled, d_disparity_map, width * height, 
                          cudaMemcpyDeviceToHost));
    
    // Convert GPU scaled disparity back to original range for comparison
    auto* disp_gpu = new float[width * height];
    for (int i = 0; i < width * height; i++) {
        disp_gpu[i] = (disp_gpu_scaled[i] * max_disparity) / 255.0f;
    }
    
    cv::Mat disp_cpu_mat(height, width, CV_32F, disp_cpu);
    cv::Mat disp_gpu_mat(height, width, CV_32F, disp_gpu);
    
    cv::Mat disp_cpu_vis, disp_gpu_vis;
    disp_cpu_mat.convertTo(disp_cpu_vis, CV_8U, 255.0 / max_disparity);
    disp_gpu_mat.convertTo(disp_gpu_vis, CV_8U, 255.0 / max_disparity);
    
    cv::Mat disp_cpu_color, disp_gpu_color;
    cv::applyColorMap(disp_cpu_vis, disp_cpu_color, cv::COLORMAP_JET);
    cv::applyColorMap(disp_gpu_vis, disp_gpu_color, cv::COLORMAP_JET);
    
    cv::Mat comparison(height, width * 2, CV_8UC3);
    disp_cpu_color.copyTo(comparison(cv::Rect(0, 0, width, height)));
    disp_gpu_color.copyTo(comparison(cv::Rect(width, 0, width, height)));
    
    cv::putText(comparison, "CPU (4 Paths)", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "GPU (4 Paths)", cv::Point(width + 10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    
    std::string output_filename = "disparity_" + dataset_name + ".png";
    cv::imwrite(output_filename, comparison);
    printf("Disparity comparison saved to: %s\n", output_filename.c_str());
    printf("  Left: CPU 4-Path | Right: GPU 4-Path\n\n");

    // ===== Summary =====
    printf("\n==============================================\n");
    printf("SUMMARY\n");
    printf("==============================================\n");
    printf("4-Path Aggregation: %s\n", passed ? "PASSED" : "FAILED");
    printf("Aggregation - CPU: %.2f ms, GPU: %.2f ms (%.2fx)\n", 
           cpu_time, gpu_time, cpu_time / gpu_time);
    printf("WTA - CPU: %.2f ms, GPU: %.2f ms (%.2fx)\n", 
           cpu_wta_time, gpu_wta_time, cpu_wta_time / gpu_wta_time);
    printf("Total - CPU: %.2f ms, GPU: %.2f ms (%.2fx)\n",
           cpu_time + cpu_wta_time, gpu_time + gpu_wta_time, 
           (cpu_time + cpu_wta_time) / (gpu_time + gpu_wta_time));
    printf("Output: %s\n", output_filename.c_str());
    printf("==============================================\n");

    // ===== Cleanup =====
    delete[] h_imL;
    delete[] h_imR;
    delete[] h_cost_volume;
    delete[] h_aggr_cpu;
    delete[] h_aggr_gpu;
    delete[] disp_cpu;
    delete[] disp_gpu;
    delete[] disp_gpu_scaled;

    cudaFree(d_imL);
    cudaFree(d_imR);
    cudaFree(d_censusL);
    cudaFree(d_censusR);
    cudaFree(d_cost_volume);
    cudaFree(d_total_cost);
    cudaFree(d_disparity_map);

    return passed ? 0 : 1;
}
