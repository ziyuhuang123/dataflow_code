import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'C:\文件备份\数据流项目\代码文件夹\my_simulate_cache\ncu_report\new\A100-flash-attention-results.csv')

# 识别Total Time (seconds)为空的行
empty_time_rows = df[df['Total Time (seconds)'].isna()]

# 提取headdim, batch_size, seqlen, dim这四列的值
headdim_batch_size_seqlen_dim_values = empty_time_rows[['headdim', 'batch_size', 'seqlen', 'dim']].values.tolist()

# 将结果保存到一个新的文件中
with open('empty_time_values_a100.txt', 'w') as f:
    for item in headdim_batch_size_seqlen_dim_values:
        f.write("%s\n" % item)

print(f"Saved {len(headdim_batch_size_seqlen_dim_values)} arrays to empty_time_values.txt")
