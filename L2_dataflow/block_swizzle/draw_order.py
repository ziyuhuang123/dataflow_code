import matplotlib.pyplot as plt
import numpy as np

def plot_compute_order_with_arrows(M, N, compute_order):
    fig, ax = plt.subplots(figsize=(N + 2, M + 2))  # 增加画布大小

    # 画出格子
    for x in range(N + 1):
        ax.axvline(x=x, color='k', linestyle='-')
    for y in range(M + 1):
        ax.axhline(y=y, color='k', linestyle='-')

    # 在格子中心画箭头
    for i in range(len(compute_order) - 1):
        start = compute_order[i]
        end = compute_order[i + 1]
        start_x, start_y = start[0] + 0.5, M - start[1] - 0.5
        end_x, end_y = end[0] + 0.5, M - end[1] - 0.5

        # 画箭头
        ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", color='r'))

    # 设置坐标轴
    ax.set_xlim(0, N)
    ax.set_ylim(0, M)
    ax.set_xticks(np.arange(0.5, N, 1))
    ax.set_yticks(np.arange(0.5, M, 1))
    ax.set_xticklabels(np.arange(1, N + 1))
    ax.set_yticklabels(np.arange(M, 0, -1))
    ax.xaxis.tick_top()

    # 添加x和y标签
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)

    plt.title('Compute Order with Arrows for Matrix C Blocks')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

M = 2
N = 6
best_order = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1), (4, 0), (5, 0), (4, 1), (5, 1)]
# [[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (5, 0), (3, 1), (4, 1), (5, 1)], [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1), (4, 0), (5, 0), (4, 1), (5, 1)]]



# 使用前面找到的最优计算顺序来画图
plot_compute_order_with_arrows(M, N, best_order)
