# 最内层横向Z，外层横向Z，整体向右移动到底。仅提供内部和外部两个Z字的宽度可选
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


# 获取所有可能的t组合
M = 6
N = 12
outer_width = 4
inner_width = 2
# result = split_to_widths(M, outer_width, inner_width)
# print(result)  # 输出: [[3, 2], [3, 2], [3, 1]]


# 针对每个t组合生成对应的order
all_orders = generate_zigzag_order_4(M, N, outer_width, inner_width)
print(all_orders)

plot_compute_order_with_arrows(M, N, all_orders)