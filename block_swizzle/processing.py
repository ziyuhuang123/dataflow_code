import pandas as pd
import matplotlib.pyplot as plt

# 读取原始CSV文件
input_file = "block_launch_times.csv"
data = pd.read_csv(input_file)

# 减去第0行的时间，得到相对时间
initial_time = data['start_time'][0]
data['start_time'] = data['start_time'] - initial_time

# 保存处理后的数据到新的CSV文件
output_file = "block_launch_times_adjusted.csv"
data.to_csv(output_file, index=False)

# 绘制图表
plt.plot(data['blockIdx.x'], data['start_time'])
plt.xlabel('blockIdx.x')
plt.ylabel('Block execution start time (clock cycles)')
plt.title('1-d block grid')
plt.show()
