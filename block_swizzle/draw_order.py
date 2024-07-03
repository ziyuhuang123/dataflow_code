import matplotlib.pyplot as plt
import numpy as np

def plot_compute_order_with_arrows(M, N, block_size, compute_order):
    # 计算每个方向上有多少个块
    blocks_m = M // block_size
    blocks_n = N // block_size

    # 创建一个足够大的画布
    fig, ax = plt.subplots(figsize=(blocks_n * 1.2, blocks_m * 1.2))

    # 画出格子
    for x in range(blocks_n + 1):
        ax.axvline(x=x, color='k', linestyle='-')
    for y in range(blocks_m + 1):
        ax.axhline(y=y, color='k', linestyle='-')

    # 在格子中心画箭头
    for i in range(len(compute_order) - 1):
        start = compute_order[i]
        end = compute_order[i + 1]
        # 转换为中心点坐标
        start_x, start_y = start[1] + 0.5, blocks_m - start[0] - 0.5
        end_x, end_y = end[1] + 0.5, blocks_m - end[0] - 0.5

        # 画箭头
        ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", color='r'))

    # 设置坐标轴
    ax.set_xlim(0, blocks_n)
    ax.set_ylim(0, blocks_m)
    ax.set_xticks(np.arange(0.5, blocks_n, 1))
    ax.set_yticks(np.arange(0.5, blocks_m, 1))
    ax.set_xticklabels(np.arange(1, blocks_n + 1))
    ax.set_yticklabels(np.arange(blocks_m, 0, -1))
    ax.xaxis.tick_top()

    plt.title('Compute Order with Arrows for Matrix C Blocks')
    plt.show()

M=512
N=512
block_size=128
best_order = [(0, 0), (0, 1), (1, 0), (1, 1),
                 (0, 2), (0, 3), (1, 2), (1, 3),
                 (2, 0), (2, 1), (3, 0), (3, 1),
                 (2, 2), (2, 3), (3, 2), (3, 3)] # 0.375MB

# 使用前面找到的最优计算顺序来画图
plot_compute_order_with_arrows(M, N, block_size, best_order)
