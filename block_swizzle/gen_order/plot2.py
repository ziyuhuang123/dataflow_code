# 尾部相连-(0,0)-(0,1)-(1,0)-(1,1)式Z型-纵向
import matplotlib.pyplot as plt
import numpy as np

def generate_zigzag_order(M, N, t):  # 将M和N互换
    order = []

    for i in range(0, N, t):  # 原来是M，现在是N
        if (i // t) % 2 == 0:
            for j in range(M):  # 原来是N，现在是M
                for k in range(t):
                    if i + k < N:  # 原来是M，现在是N
                        order.append((i + k, j))
        else:
            for j in range(M - 1, -1, -1):  # 原来是N，现在是M
                for k in range(t):
                    if i + k < N:  # 原来是M，现在是N
                        order.append((i + k, j))

    return order


def plot_compute_order_with_arrows(M, N, compute_order):
    fig, ax = plt.subplots(figsize=(N, M))

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

    plt.title('Compute Order with Arrows for Matrix C Blocks')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

M = 6 # 行数
N = 6 # 列数
t = 3

order = generate_zigzag_order(M, N, t)
print(order)
plot_compute_order_with_arrows(M, N, order)


# 明天继续修改为提供两种Z宽度的选择