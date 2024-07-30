import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/weile/project/Hopper-Microbenchmark/NewFeatures/DSM/block_sm.csv', header=None)

# 获取第二列的唯一值的数量
unique_values_count = df.iloc[:, 1].nunique()

print(f"第二列有 {unique_values_count} 个不同的数字。")

# 按照第一列升序排序
df_sorted = df.sort_values(by=df.columns[0], ascending=True)

df_sorted.to_csv('/home/weile/project/Hopper-Microbenchmark/NewFeatures/DSM/sorted_file.csv', header=None, index=False)


# 获取第二列，因为没有列标题，所以第二列的索引是 1
second_column = df_sorted.iloc[:, 1]

# 创建一个标记是否有重复的变量
has_duplicates = False

n = 2
# 检查每组n个值中是否有重复
for i in range(0, len(second_column), n):
    # 取出当前的n个值
    current_values = second_column[i:i+n]
    # 检查当前n个值是否有重复
    if current_values.nunique() != len(current_values):
        has_duplicates = True
        print(f"从索引 {i} 到 {i+n} 的四个值中有重复: {current_values.tolist()}")
        break  # 如果找到重复的，就跳出循环

# 如果没有重复的，打印消息
if not has_duplicates:
    print(f"在第二列中每{n}个值中没有找到重复的数字。")