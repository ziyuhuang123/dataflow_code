# 定义矩阵维度和 block 大小
M, N, K = 512, 512, 128
block_size = 128
data_type_size = 2  # 'half' 类型数据的大小为 2 字节

# 计算每个 block 的数据量（字节）
block_data_size_bytes = block_size * block_size * data_type_size

# L2 缓存大小和容量
L2_cache_size_kb = 128
# 能装得下128*128的块多少个
L2_cache_capacity = (L2_cache_size_kb * 1024) // block_data_size_bytes

# 初始化 L2 缓存和全局内存读取次数
L2_cache = []
global_memory_reads = {'A': 0, 'B': 0}


def access_memory(block_idx, needed_blocks_A, needed_blocks_B):
    global L2_cache, global_memory_reads

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
                evicted = evict_candidates[0] if evict_candidates else L2_cache[0]  # 若所有块即将使用，选择最旧的
                L2_cache.remove(evicted)
                print(f"从 L2 挤出: {evicted}")

            L2_cache.append(full_block_id)
            global_memory_reads[matrix_label] += 1
            print(f"从 global 读取: {full_block_id}")
        else:
            print(f"从 L2 读取: {full_block_id}")

    # 访问 A 和 B 矩阵块
    print(f"\n正在计算 C 矩阵的 block: ({block_idx[0]}, {block_idx[1]})")
    print("访问 A 矩阵:")
    for block in needed_blocks_A:
        check_and_update_cache('A', block)

    print("访问 B 矩阵:")
    for block in needed_blocks_B:
        check_and_update_cache('B', block)

    print(f"当前 L2 缓存的块: {L2_cache}")

# 确定 block 计算顺序
compute_order = [(i, j) for i in range(M // block_size) for j in range(N // block_size)] # 0.625MB
# compute_order = [(0, 0), (0, 1), (1, 0), (1, 1),
#                  (0, 2), (0, 3), (1, 2), (1, 3),
#                  (2, 0), (2, 1), (3, 0), (3, 1),
#                  (2, 2), (2, 3), (3, 2), (3, 3)] # 0.375MB
# compute_order = [(0, 0), (0, 1), (1, 0), (1, 1),
#                  (2, 0), (2, 1), (3, 0), (3, 1),
#                  (0, 2), (0, 3), (1, 2), (1, 3),
#                  (2, 2), (2, 3), (3, 2), (3, 3)] # 0.4375MB
# 模拟每个 block 的计算，并打印访问信息
for block_idx in compute_order:
    needed_blocks_A = [(block_idx[0], k) for k in range(K // block_size)]
    needed_blocks_B = [(k, block_idx[1]) for k in range(K // block_size)]
    access_memory(block_idx, needed_blocks_A, needed_blocks_B)

# 计算全局内存加载的总数据量（MB）
total_data_bytes = (global_memory_reads['A'] + global_memory_reads['B']) * block_data_size_bytes
total_data_mb = total_data_bytes / (1024 * 1024)

print(f"\n全局内存加载的总数据量（MB）: {total_data_mb:.6f} MB")