import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'C:\文件备份\数据流项目\代码文件夹\my_simulate_cache\ncu_report\new\results_a40_FFN1.csv')

# 删除所有 Performance Bound 列中值为 "compute bound" 的行
df.loc[df['Performance Bound'] == 'compute bound', 'Performance Bound'] = ''

# 定义组的大小
group_size = 7

# 计算每组内的base_ratio
for i in range(0, len(df), group_size):
    group = df.iloc[i:i + group_size]
    if len(group) == group_size:
        base_time = group['Total Time (seconds)'].iloc[0]
        df.loc[i:i + group_size - 1, 'BaselineRatio'] = base_time / group['Total Time (seconds)']

# 将结果写回CSV文件
df.to_csv(r'C:\文件备份\数据流项目\代码文件夹\my_simulate_cache\ncu_report\new\results_a40_FFN1.csv', index=False)

print("Base ratios calculated and saved to results.csv.")
