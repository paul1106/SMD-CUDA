#!/bin/bash

# Baseline Benchmark Script for SGM CPU Implementation
# This script runs all datasets and records timing for comparison with CUDA version

OUTPUT_FILE="baseline_results.txt"

echo "=========================================" > $OUTPUT_FILE
echo "SGM CPU Baseline Benchmark Results" >> $OUTPUT_FILE
echo "Date: $(date)" >> $OUTPUT_FILE
echo "=========================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Function to run a single test
run_test() {
    local dataset=$1
    local left_img=$2
    local right_img=$3
    local min_disp=$4
    local max_disp=$5
    
    echo "========================================="
    echo "Testing: $dataset"
    echo "Resolution: $(identify -format "%wx%h" $left_img 2>/dev/null || echo "unknown")"
    echo "Disparity range: [$min_disp, $max_disp]"
    echo "========================================="
    
    echo "----------------------------------------" >> $OUTPUT_FILE
    echo "Dataset: $dataset" >> $OUTPUT_FILE
    
    # Get image resolution
    local resolution=$(identify -format "%wx%h" $left_img 2>/dev/null || echo "unknown")
    echo "Resolution: $resolution" >> $OUTPUT_FILE
    echo "Disparity Range: [$min_disp, $max_disp]" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    
    # Run SGM and capture output
    output=$(bin/sgm_cpu $left_img $right_img $min_disp $max_disp 2>&1)
    
    # Extract timing information and convert to milliseconds
    init_time=$(echo "$output" | grep "SGM Initializing Done" | grep -oP '\d+\.\d+' | awk '{printf "%.2f", $1*1000}')
    cost_time=$(echo "$output" | grep "computing cost" | grep -oP '\d+\.\d+' | awk '{printf "%.2f", $1*1000}')
    aggr_time=$(echo "$output" | grep "cost aggregating" | grep -oP '\d+\.\d+' | awk '{printf "%.2f", $1*1000}')
    disp_time=$(echo "$output" | grep "computing disparities" | grep -oP '\d+\.\d+' | awk '{printf "%.2f", $1*1000}')
    post_time=$(echo "$output" | grep "postprocessing" | grep -oP '\d+\.\d+' | awk '{printf "%.2f", $1*1000}')
    total_time=$(echo "$output" | grep "SGM Matching...Done" | grep -oP '\d+\.\d+' | awk '{printf "%.2f", $1*1000}')
    
    # Print results
    echo "Initialization:       ${init_time} ms" | tee -a $OUTPUT_FILE
    echo "Cost Computation:     ${cost_time} ms" | tee -a $OUTPUT_FILE
    echo "Cost Aggregation:     ${aggr_time} ms" | tee -a $OUTPUT_FILE
    echo "Disparity Calc:       ${disp_time} ms" | tee -a $OUTPUT_FILE
    echo "Post-processing:      ${post_time} ms" | tee -a $OUTPUT_FILE
    echo "Total Matching Time:  ${total_time} ms" | tee -a $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
}

# Run all datasets
run_test "cone" "Data/cone/im2.png" "Data/cone/im6.png" 0 64
run_test "Reindeer" "Data/Reindeer/view1.png" "Data/Reindeer/view5.png" 0 128
run_test "Cloth3" "Data/Cloth3/view1.png" "Data/Cloth3/view5.png" 0 128
run_test "Wood2" "Data/Wood2/view1.png" "Data/Wood2/view5.png" 0 128

echo "========================================="
echo "Benchmark completed! Results saved to: $OUTPUT_FILE"
echo "========================================="

# Display summary
echo "" >> $OUTPUT_FILE
echo "=========================================" >> $OUTPUT_FILE
echo "Summary" >> $OUTPUT_FILE
echo "=========================================" >> $OUTPUT_FILE
echo "All timing measurements are in milliseconds (ms)" >> $OUTPUT_FILE
echo "These results serve as CPU baseline for CUDA optimization comparison" >> $OUTPUT_FILE
