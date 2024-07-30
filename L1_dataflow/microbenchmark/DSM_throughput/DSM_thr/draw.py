import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('results.csv')

# 将数据分割为两部分：Block Size = 128 和 Block Size = 512
data_128 = data[data['Block Size'] == 128]
data_512 = data[data['Block Size'] == 512]

# 定义绘图函数
def plot_throughput(data, block_size, ax):
    nbins_values = sorted(data['Nbins'].unique())
    cluster_size_values = sorted(data['Cluster Size'].unique())
    
    bar_width = 0.15
    colors = ['r', 'g', 'b', 'y', 'c']

    for i, cluster_size in enumerate(cluster_size_values):
        subset = data[data['Cluster Size'] == cluster_size]
        ax.bar(
            [x + i * bar_width for x in range(len(nbins_values))],
            subset['Throughput (Gelem/s)'],
            width=bar_width,
            label=f'CS = {cluster_size}',
            color=colors[i % len(colors)]
        )
    
    ax.set_xlabel('Nbins')
    ax.set_ylabel('Throughput ($10^9$ element/s)')
    ax.set_title(f'Block Size = {block_size}')
    ax.set_xticks([r + bar_width * (len(cluster_size_values) - 1) / 2 for r in range(len(nbins_values))])
    ax.set_xticklabels(nbins_values)
    ax.legend()

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# 绘制Block Size = 128的图表
plot_throughput(data_128, 128, ax1)

# 绘制Block Size = 512的图表
plot_throughput(data_512, 512, ax2)

plt.tight_layout()
plt.savefig('throughput_comparison.png')
plt.show()
