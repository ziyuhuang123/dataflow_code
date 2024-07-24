import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 数据字典
data_A100 = {
    "Lamma3-70B-A100-FFN0": [66.87, 83.31, 84.23, 86.6, 87.3, 87.16, 86.95, 84.89, 84.79],
    "Lamma3-70B-A100-FFN1": [None, 91.22, 91.02, None, None, 91.14, 90.93, 91.04, None],
    "Lamma3-8B-A100-FFN0": [66.28, 83.62, 85.31, 87.31, 88.1, 88.9, 89.08, 89.41, None],
    "Lamma3-8B-A100-FFN1": [None, 94.06, 95.86, None, None, 94.03, 95.92, 90.97, None]
}

# X轴数据
x = [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
x_log2 = np.log2(x)  # 将X轴数据转换为以2为底的对数

# 创建图表
plt.figure(figsize=(14, 8))

# 画每一条线
for label, y in data_A100.items():
    if "70B" in label:
        plt.plot(x_log2, y, marker='o', linestyle='-', linewidth=2, label=label)  # 粗线和圆点
    elif "8B" in label:
        plt.plot(x_log2, y, marker='^', linestyle='-', linewidth=1, label=label)  # 细线和三角形点

# 设置图表标题和标签
plt.title('L2 Hit Rate for A100 Configurations')
plt.xlabel('log2(M)')
plt.ylabel('L2 Hit Rate (%)')

# 设置X轴刻度为以2为底的对数刻度
plt.xticks(x_log2, x)

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
