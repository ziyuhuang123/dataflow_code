# 尾部相连-(0,0)-(0,1)-(1,0)-(1,1)式Z型-纵向
# import matplotlib.pyplot as plt
import numpy as np


# def plot_compute_order_with_arrows(M, N, compute_order):
#     fig, ax = plt.subplots(figsize=(N + 2, M + 2))  # 增加画布大小

#     # 画出格子
#     for x in range(N + 1):
#         ax.axvline(x=x, color='k', linestyle='-')
#     for y in range(M + 1):
#         ax.axhline(y=y, color='k', linestyle='-')

#     # 在格子中心画箭头
#     for i in range(len(compute_order) - 1):
#         start = compute_order[i]
#         end = compute_order[i + 1]
#         start_x, start_y = start[0] + 0.5, M - start[1] - 0.5
#         end_x, end_y = end[0] + 0.5, M - end[1] - 0.5

#         # 画箭头
#         ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
#                     arrowprops=dict(arrowstyle="->", color='r'))

#     # 设置坐标轴
#     ax.set_xlim(0, N)
#     ax.set_ylim(0, M)
#     ax.set_xticks(np.arange(0.5, N, 1))
#     ax.set_yticks(np.arange(0.5, M, 1))
#     ax.set_xticklabels(np.arange(1, N + 1))
#     ax.set_yticklabels(np.arange(M, 0, -1))
#     ax.xaxis.tick_top()

#     # 添加x和y标签
#     ax.set_xlabel('x', fontsize=14)
#     ax.set_ylabel('y', fontsize=14)

#     plt.title('Compute Order with Arrows for Matrix C Blocks')
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.show()

def generate_zigzag_order_1(M, N, T1):
    def find_combination(N, t1):
        combination = []
        while N > 0:
            if N >= t1:
                combination.append(t1)
                N -= t1
            else:
                combination.append(N)
                break
        return combination

    t_combo = find_combination(N, T1)

    orders = []
    current_x_index = 0
    tail_connect_flag = False
    for t in t_combo:
        if tail_connect_flag == 0:
            for j in range(M):
                for k in range(t):
                    if current_x_index < N:
                        orders.append((current_x_index + k, j))
            current_x_index += t
            tail_connect_flag = not tail_connect_flag
        elif tail_connect_flag == 1:
            for j in range(M - 1, -1, -1):
                for k in range(t):
                    if current_x_index < N:
                        orders.append((current_x_index + k, j))
            current_x_index += t
            tail_connect_flag = not tail_connect_flag
    return orders

# 示例
M = 9
N = 6
t1 = 4
# t2 = 3

# 获取所有可能的t组合
# combinations = find_combinations(N, t1, t2)
# print(combinations)
# 针对每个t组合生成对应的order

all_orders = generate_zigzag_order_1(M, N, t1)

print(all_orders)
# plot_compute_order_with_arrows(M, N, all_orders)