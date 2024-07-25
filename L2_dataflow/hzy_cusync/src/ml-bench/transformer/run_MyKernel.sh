#!/bin/bash

# Directory to save output files
# output_dir="/home/zyhuang/temp_can/dataflow_code/block_swizzle/gen_two_order"

output_dir="/root/zyhuang/gen_two_order"
# Output CSV file
output_csv="Mykernel_output_results_SQZ.csv"

# Initialize CSV file with headers
echo "batch,order_line,average_time" > ${output_csv}

# Loop through batch sizes from 2^8 to 2^15
for ((i=8; i<=15; i++)); do
    batch=$((2**i))
    
    # Loop through order_line values from 1 to 6
    for order_line in {1..6}; do
        # Construct the file path with the corresponding batch size
        file_path="${output_dir}/order_${batch}.txt"
        
        # Construct the command
        cmd="build/mlp-eval-rowsync --batch ${batch} --check true --model gpt3 --split-k1 1 --split-k2 1 --policy cusync --order-line ${order_line} --file-path ${file_path}"
        echo "Running: ${cmd}"
        
        # Run the command and capture the output
        output=$(eval "${cmd}")
        
        # Extract the average time from the output
        avg_time=$(echo "${output}" | grep "Average time" | awk '{print $3}')
        
        # Write the results to the CSV file
        echo "${batch},${order_line},${avg_time}" >> ${output_csv}
    done
done
