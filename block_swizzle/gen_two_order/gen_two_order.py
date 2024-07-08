# 单独Z型-(0,0)-(0,1)-(1,0)-(1,1)式Z型-横向-->生成两个GEMM交叉的顺序
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm

def generate_level1Z_order(M, N, t, execute_order):
    order = []
    for i in range(0, M, t):
        for j in range(N):
            for k in range(t):
                if i + k < M:
                    order.append((j, i + k, execute_order))
    return order


def generate_zigzag_order(M, N0, N1, t):

    order0 = generate_level1Z_order(M, N0, t, 0)
    order1 = generate_level1Z_order(M, N1, t, 1)
    new_order = []
    in_order_0 = True
    current_y_loc = 0
    while(len(new_order)<M*(N0+N1)):
        if in_order_0==True:
            for i in range(t*N0):
                new_order.append(order0[i+current_y_loc*N0])
            in_order_0 = not in_order_0
        else:
            for j in range(t*N1):
                new_order.append(order1[j+current_y_loc*N1])
            in_order_0 = not in_order_0
            current_y_loc += t
    return new_order


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

def main():
    parser = argparse.ArgumentParser(description='Generate zigzag order')
    parser.add_argument('--M', type=int, required=True, help='Number of rows')
    parser.add_argument('--t', type=int, required=True, help='Tiling size')
    args = parser.parse_args()
    
    M = args.M
    t = args.t
    N0 = 5  # 固定列数
    N1 = 4  # 固定列数

    order = generate_zigzag_order(M, N0, N1, t)
    
    # print(order)

if __name__ == "__main__":
    main()