def calculate_total_data_mb(M, N, K, block_size, data_type_size, L2_cache_size_kb, compute_order):
    # 计算每个 block 的数据量（字节）
    block_data_size_bytes = block_size * block_size * data_type_size

    # L2 缓存容量
    L2_cache_capacity = (L2_cache_size_kb * 1024) // block_data_size_bytes

    # 初始化 L2 缓存和全局内存读取次数
    L2_cache = []
    global_memory_reads = {'A': 0, 'B': 0}

    def access_memory(block_idx, needed_blocks_A, needed_blocks_B):
        nonlocal L2_cache, global_memory_reads

        # 计算即将需要使用的块集合
        needed_blocks = {(matrix_label, block) for matrix_label, blocks in [('A', needed_blocks_A), ('B', needed_blocks_B)] for block in blocks}

        # 检查和更新缓存的函数
        def check_and_update_cache(matrix_label, block):
            full_block_id = (matrix_label, block)

            # 如果块不在缓存中
            if full_block_id not in L2_cache:
                # 如果缓存已满，尝试找到一个非即将使用的块来驱逐
                if len(L2_cache) >= L2_cache_capacity:
                    # 选择非即将使用的块驱逐
                    evict_candidates = [blk for blk in L2_cache if blk not in needed_blocks]
                    evicted = evict_candidates[0] if evict_candidates else L2_cache[0]
                    L2_cache.remove(evicted)

                L2_cache.append(full_block_id)
                global_memory_reads[matrix_label] += 1

        # 访问 A 和 B 矩阵块
        for block in needed_blocks_A:
            check_and_update_cache('A', block)

        for block in needed_blocks_B:
            check_and_update_cache('B', block)

    # 模拟每个 block 的计算
    for block_idx in compute_order:
        needed_blocks_A = [(block_idx[0], k) for k in range(K // block_size)]
        needed_blocks_B = [(k, block_idx[1]) for k in range(K // block_size)]
        access_memory(block_idx, needed_blocks_A, needed_blocks_B)

    # 计算全局内存加载的总数据量（MB）
    total_data_bytes = (global_memory_reads['A'] + global_memory_reads['B']) * block_data_size_bytes
    total_data_mb = total_data_bytes / (1024 * 1024)

    return total_data_mb

# 输入参数
M, N, K = 384, 384, 128
block_size = 128
data_type_size = 2
L2_cache_size_kb = 128

# 计算顺序示例
compute_order = [(0, 1), (1, 0), (0, 0), (1, 1), (0, 2), (2, 0), (2, 2), (1, 2), (2, 1)]
# compute_order = [(i, j) for i in range(M // block_size) for j in range(N // block_size)] # 0.625MB
# 计算 total_data_mb
total_data_mb = calculate_total_data_mb(M, N, K, block_size, data_type_size, L2_cache_size_kb, compute_order)
print(f"全局内存加载的总数据量（MB）: {total_data_mb:.6f} MB")
