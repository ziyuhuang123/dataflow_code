# 单独Z型-(0,0)-(0,1)-(1,0)-(1,1)式Z型-纵向-提供两种Z宽度可选
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

def generate_zigzag_order(M, N, T1, T2):
    def find_combinations(N, t1, t2):
        combinations = []
        for i in range(N // t1 + 1):
            for j in range(N // t2 + 1):
                if i * t1 + j * t2 == N:
                    combinations.append([t1] * i + [t2] * j)
        return combinations

    t_combo = find_combinations(N, T1, T2)
    orders = []
    for t_set in t_combo:
        order = []
        current_x_index = 0
        for t in t_set:
            for j in range(M):
                for k in range(t):
                    if current_x_index < N:
                        order.append((current_x_index+k, j))
            current_x_index += t
        orders.append(order)
    return orders

# 示例
M = 2
N = 6
t1 = 2
t2 = 3

# 获取所有可能的t组合
# combinations = find_combinations(N, t1, t2)
# print(combinations)
# 针对每个t组合生成对应的order

all_orders = generate_zigzag_order(M, N, t1, t2)

print(all_orders)
for one_order in all_orders:
    plot_compute_order_with_arrows(M, N, one_order)