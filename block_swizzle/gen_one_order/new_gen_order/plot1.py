# 单独Z型-(0,0)-(0,1)-(1,0)-(1,1)式Z型-纵向
# import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
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

def generate_zigzag_order_0(M, N, T1):
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
    
    for t in t_combo:
        for j in range(M):
            for k in range(t):
                if current_x_index < N:
                    orders.append((current_x_index + k, j))
        current_x_index += t
    
    return orders

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

def generate_zigzag_order_2(M, N, T1):
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

    t_combo = find_combination(M, T1)
    orders = []
    current_x_index = 0
    for t in t_combo:
        for j in range(N):
            for k in range(t):
                if current_x_index < M:
                    orders.append((j, current_x_index + k))
        current_x_index += t
    return orders

def generate_zigzag_order_3(M, N, T1):
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

    t_combo = find_combination(M, T1)
    orders = []
    current_x_index = 0
    tail_connect_flag = False
    for t in t_combo:
        if tail_connect_flag == 0:
            for j in range(N):
                for k in range(t):
                    if current_x_index < M:
                        orders.append((j, current_x_index + k))
            current_x_index += t
            tail_connect_flag = not tail_connect_flag
        elif tail_connect_flag == 1:
            for j in range(N - 1, -1, -1):
                for k in range(t):
                    if current_x_index < M:
                        orders.append((j, current_x_index + k))
            current_x_index += t
            tail_connect_flag = not tail_connect_flag
    return orders

def generate_zigzag_order_4(M, N, Outer_width, Inner_width):
    def split_to_widths(M, outer_width, inner_width):
        if outer_width < inner_width:
            raise ValueError("outer_width must be greater than or equal to inner_width")

        # 对外层宽度进行拆分
        outer_splits = []
        while M > 0:
            if M >= outer_width:
                outer_splits.append(outer_width)
                M -= outer_width
            else:
                outer_splits.append(M)
                break

        # 对内部宽度进行拆分，并将结果作为子列表添加到最终结果列表中
        result = []
        for width in outer_splits:
            inner_result = []
            while width > 0:
                if width >= inner_width:
                    inner_result.append(inner_width)
                    width -= inner_width
                else:
                    inner_result.append(width)
                    break
            result.append(inner_result)

        return result

    width_list = split_to_widths(M, Outer_width, Inner_width)

    orders = []
    big_yaxis_loc = 0
    for inner_width_list in width_list:
        current_xaxis_loc = 0
        for num_iter in range(int((N+1)/2)):
            current_yaxis_loc = 0
            for width in inner_width_list:
                for j in range(2):
                    for k in range(width):
                        if j+current_xaxis_loc<N:
                            orders.append((j+current_xaxis_loc, k+current_yaxis_loc+big_yaxis_loc))
                current_yaxis_loc += width
            current_xaxis_loc += 2
        for t in inner_width_list:
            big_yaxis_loc += t
    return orders


def write_orders_to_file(f, description, orders):
    f.write(f'{description} [')
    f.write(', '.join([f'({x}, {y}, 0)' for x, y in orders]))
    f.write(']\n')


def main(M):
    N = 112

    # 遍历 t1 从 2^1 到 2^7，并将所有 order 写入文件
    with open(f'orders_{M}.txt', 'w') as f:
        # 遍历 t1 从 2^1 到 2^7
        for i in tqdm(range(0, 6), desc="Processing t1 values and generating orders"):
            t1 = 2 ** i
            
            all_orders_0 = generate_zigzag_order_0(M, N, t1)
            write_orders_to_file(f, f't1 = {t1} Order 0:', all_orders_0)
            
            all_orders_1 = generate_zigzag_order_1(M, N, t1)
            write_orders_to_file(f, f't1 = {t1} Order 1:', all_orders_1)
            
            all_orders_2 = generate_zigzag_order_2(M, N, t1)
            write_orders_to_file(f, f't1 = {t1} Order 2:', all_orders_2)
            
            all_orders_3 = generate_zigzag_order_3(M, N, t1)
            write_orders_to_file(f, f't1 = {t1} Order 3:', all_orders_3)

        # 注意这里的i不能从0开始，因为inner_width不能为2^-1
        for i in tqdm(range(1, 6), desc="Processing outer_width and inner_width values and generating orders"):
            outer_width = 2 ** i
            inner_width = 2 ** (i - 1)
            
            all_orders_4 = generate_zigzag_order_4(M, N, outer_width, inner_width)
            write_orders_to_file(f, f'outer_width = {outer_width}, inner_width = {inner_width} Order 4:', all_orders_4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate orders for given M and N')
    parser.add_argument('--M', type=int, required=True, help='Value of M')
    args = parser.parse_args()
    real_block_M = args.M/128 # 注意这里强行指定了block维度是128！！！以后可能需要修改！！
    main(args.M)