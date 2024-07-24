import itertools
import numpy as np
from tqdm import tqdm
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import time

# 定义几何变换
def transform_coordinates(order, transform):
    return [transform(x, y) for x, y in order]

def rotate_90(x, y):
    return (y, 3 - x)

def rotate_180(x, y):
    return (3 - x, 3 - y)

def rotate_270(x, y):
    return (3 - y, x)

def mirror_horizontal(x, y):
    return (x, 3 - y)

def mirror_vertical(x, y):
    return (3 - x, y)

def mirror_diagonal(x, y):
    return (y, x)

def mirror_anti_diagonal(x, y):
    return (3 - y, 3 - x)

def normalize_order(order):
    transforms = [
        lambda x, y: (x, y),
        rotate_90,
        rotate_180,
        rotate_270,
        mirror_horizontal,
        mirror_vertical,
        mirror_diagonal,
        mirror_anti_diagonal
    ]

    normalized_orders = []
    for transform in transforms:
        transformed_order = transform_coordinates(order, transform)
        normalized_orders.append(tuple(transformed_order))

    return min(normalized_orders)

def process_orders_batch(orders):
    unique_orders = set()
    for order in orders:
        normalized_order = normalize_order(order)
        unique_orders.add(normalized_order)
    return unique_orders

def save_optimal_order_to_file(order, access_mb, folder):
    timestamp = int(time.time() * 1000)
    filename = f"{folder}/order_{timestamp}.txt"
    with open(filename, 'w') as f:
        compute_order_formatted = ", ".join([f"({x}, {y})" for x, y in order])
        f.write(f"compute order: [{compute_order_formatted}], global memory access: {access_mb:.6f} MB\n")

def generate_and_evaluate_unique_compute_orders(batch_size=10000, output_file='unique_compute_orders.pkl', result_folder='results'):
    coordinates = [(x, y) for x in range(4) for y in range(4)]
    all_orders = itertools.permutations(coordinates)
    unique_orders = set()

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, 'rb') as f:
                unique_orders = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print("Failed to load existing unique orders. Starting from scratch.")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    optimal_orders = []
    min_access = 0.4  # Set the initial min access to 0.4 to filter only better results

    with ProcessPoolExecutor() as executor:
        batch = []
        for i, order in enumerate(tqdm(all_orders, desc="Generating unique compute orders")):
            batch.append(order)
            if len(batch) == batch_size:
                batch_unique_orders = executor.submit(process_orders_batch, batch).result()
                batch = []

                # Evaluate the unique orders batch
                for unique_order in batch_unique_orders:
                    access_mb = evaluate_order(unique_order)
                    if access_mb < min_access:
                        min_access = access_mb
                        optimal_orders.append((unique_order, access_mb))
                        print(f"Order with memory access < 0.4MB: {unique_order} -> {access_mb:.6f} MB")
                        save_optimal_order_to_file(unique_order, access_mb, result_folder)

                unique_orders.update(batch_unique_orders)
                with open(output_file, 'wb') as f:
                    pickle.dump(unique_orders, f)
                print(f"Stored {len(unique_orders)} unique orders so far.")

                # 检查是否找到足够多的小于0.4MB的结果
                if len(optimal_orders) >= 10000:  # 可以根据需要调整
                    break

        # 处理最后一批次
        if batch:
            batch_unique_orders = executor.submit(process_orders_batch, batch).result()
            for unique_order in batch_unique_orders:
                access_mb = evaluate_order(unique_order)
                if access_mb < min_access:
                    min_access = access_mb
                    optimal_orders.append((unique_order, access_mb))
                    print(f"Order with memory access < 0.4MB: {unique_order} -> {access_mb:.6f} MB")
                    save_optimal_order_to_file(unique_order, access_mb, result_folder)

            unique_orders.update(batch_unique_orders)
            with open(output_file, 'wb') as f:
                pickle.dump(unique_orders, f)
            print(f"Stored {len(unique_orders)} unique orders so far.")

    return optimal_orders

def evaluate_order(order):
    M, N, K = 512, 512, 128
    block_size = 128
    data_type_size = 2
    L2_cache_size_kb = 128
    block_data_size_bytes = block_size * block_size * data_type_size
    L2_cache_capacity = (L2_cache_size_kb * 1024) // block_data_size_bytes

    L2_cache = []
    global_memory_reads_A = 0
    global_memory_reads_B = 0

    for block_idx in order:
        needed_blocks_A = [(block_idx[0], k) for k in range(K // block_size)]
        needed_blocks_B = [(k, block_idx[1]) for k in range(K // block_size)]

        needed_blocks = []
        for j in range(len(needed_blocks_A)):
            needed_blocks.append(('A', needed_blocks_A[j]))
        for j in range(len(needed_blocks_B)):
            needed_blocks.append(('B', needed_blocks_B[j]))

        def check_and_update_cache(matrix_label, block):
            nonlocal global_memory_reads_A, global_memory_reads_B
            full_block_id = (matrix_label, block)
            if full_block_id not in L2_cache:
                if len(L2_cache) >= L2_cache_capacity:
                    evict_candidates = []
                    for blk in L2_cache:
                        if blk not in needed_blocks:
                            evict_candidates.append(blk)
                    evicted = evict_candidates[0] if len(evict_candidates) > 0 else L2_cache[0]
                    L2_cache.remove(evicted)
                L2_cache.append(full_block_id)
                if matrix_label == 'A':
                    global_memory_reads_A += 1
                else:
                    global_memory_reads_B += 1

        for j in range(len(needed_blocks_A)):
            check_and_update_cache('A', needed_blocks_A[j])
        for j in range(len(needed_blocks_B)):
            check_and_update_cache('B', needed_blocks_B[j])

    total_data_bytes = (global_memory_reads_A + global_memory_reads_B) * block_data_size_bytes
    total_data_mb = total_data_bytes / (1024 * 1024)

    return total_data_mb

if __name__ == '__main__':
    optimal_orders = generate_and_evaluate_unique_compute_orders(batch_size=10000)
    print("Optimal orders with memory access < 0.4MB:")
    for order, access in optimal_orders:
        print(f"Order: {order}, Memory access: {access:.6f} MB")

    with open("optimal_compute_orders.txt", "w") as f:
        f.write(f"全局内存加载量最小的compute order和其加载量:\n")
        for order, access in optimal_orders:
            compute_order_formatted = ", ".join([f"({x}, {y})" for x, y in order])
            f.write(f"compute order: [{compute_order_formatted}], global memory access: {access:.6f} MB\n")
            print(f"compute order: [{compute_order_formatted}], global memory access: {access:.6f} MB")
        f.write(f"\n最小的全局内存访问量（MB）: {min(access for _, access in optimal_orders):.6f} MB\n")
        print(f"\n最小的全局内存访问量（MB）: {min(access for _, access in optimal_orders):.6f} MB")

    print("Process completed.")
