# 下面这些是Identity的策略。如果是horizontal就没有任何Z字形，就一次算完一整行，然后再下一行那样。
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def gemm_transform(block_idx_x, block_idx_y, log_tile):
    transformed_x = block_idx_x >> log_tile
    transformed_y = (block_idx_y << log_tile) + (block_idx_x & ((1 << log_tile) - 1))
    return transformed_x, transformed_y

def print_transformation(log_tile, x_max, y_max):
    transformations = []
    for block_idx_x in range(x_max):
        for block_idx_y in range(y_max):
            tx, ty = gemm_transform(block_idx_x, block_idx_y, log_tile)
            transformations.append(((block_idx_x, block_idx_y), (tx, ty)))
            print(f"({block_idx_x}, {block_idx_y}) ---> ({tx}, {ty})")
    return transformations

def plot_grid_with_transformations(transformations, x_useful_max, y_useful_max):
    fig, ax = plt.subplots(figsize=(10, 10))

    # 计算变换后的最大坐标
    max_x_trans = max(trans[0] for _, trans in transformations)
    max_y_trans = max(trans[1] for _, trans in transformations)

    # 设定网格尺寸
    max_x = max_x_trans + 1
    max_y = max_y_trans + 1

    # 创建网格
    for i in range(max_x + 1):
        ax.axvline(i, color='gray', linewidth=0.5)
    for j in range(max_y + 1):
        ax.axhline(j, color='gray', linewidth=0.5)

    # 标注变换前后的索引，并标记黄色区域
    for (orig, trans) in transformations:
        orig_x, orig_y = orig
        trans_x, trans_y = trans
        if trans_x < x_useful_max and trans_y < y_useful_max:
            ax.add_patch(plt.Rectangle((trans_x, trans_y), 1, 1, color='yellow', alpha=0.3))
        ax.text(trans_x + 0.5, trans_y + 0.5, f"({orig_x}, {orig_y})", ha='center', va='center', fontsize=8, color='blue')

    # 添加标签
    ax.text(max_x / 2, -0.5, "黄色区域：真正执行的区域，其他区域：被if筛掉而直接return。（a,b）表示idx.x=a，idx.y=b", ha='center', va='center', fontsize=12, color='red')

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.title('Grid with Transformed Indices')
    plt.xlabel('Transformed blockIdx.x')
    plt.ylabel('Transformed blockIdx.y')
    plt.show()

# 示例使用
log_tile = 1
x_useful_max = 5
y_useful_max = 5

x_max = x_useful_max* (1<<log_tile)
y_max = int((y_useful_max  + (1<<log_tile) - 1) / (1<<log_tile))

transformations = print_transformation(log_tile, x_max, y_max)
plot_grid_with_transformations(transformations, x_useful_max, y_useful_max)
