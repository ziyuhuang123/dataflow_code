import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
data = pd.read_csv('results.csv')

# 设置图表大小
plt.figure(figsize=(12, 8))

# 获取所有的cluster_sizes和threads_per_blocks
cluster_sizes = data['Cluster Size'].unique()
threads_per_blocks = data['Threads Per Block'].unique()

# 定义颜色列表用于不同的Cluster Size
colors = plt.cm.tab20.colors

# 绘制柱状图
bar_width = 0.1  # 设置柱状图宽度
bar_positions = np.arange(len(threads_per_blocks))
x_positions = bar_positions * (len(cluster_sizes) + 1) * bar_width  # 等距排列位置
for i, cluster_size in enumerate(cluster_sizes):
    cluster_data = data[data['Cluster Size'] == cluster_size]
    positions = x_positions + i * bar_width
    plt.bar(positions, cluster_data['Throughput (TB/s)'], width=bar_width, label=f'Cluster Size {cluster_size}', color=colors[i % len(colors)])

# 设置图表标题和标签
plt.title('Throughput vs Threads Per Block for Different Cluster Sizes')
plt.xlabel('Threads Per Block')
plt.ylabel('Throughput (TB/s)')
plt.yscale('log')  # 设置纵坐标为对数刻度
plt.xticks(x_positions + bar_width * (len(cluster_sizes) // 2), threads_per_blocks)  # 设置横坐标刻度
plt.legend()

# 保存图表
plt.tight_layout()
plt.savefig('throughput_vs_threads_per_block_log_y.png')
plt.show()
