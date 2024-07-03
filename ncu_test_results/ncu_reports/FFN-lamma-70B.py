import torch

# 设置参数
b = 8  # batch size
s = 8192  # sequence length
h = 8192  # hidden size
intermediate_size = 28672  # intermediate size for FFN layers
dtype = torch.half  # 使用 half 精度

# 检查是否有可用的 GPU 并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取当前设备的总显存量
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
    print(f"Total GPU memory: {total_memory:.2f} MB")
else:
    print("No GPU available")

# 初始化计时器和内存统计
torch.cuda.reset_peak_memory_stats(device)

# 初始化矩阵并转换为 half precision 并移动到 GPU
A = torch.randn(b * s, h, dtype=dtype, device=device)
B = torch.randn(h, intermediate_size, dtype=dtype, device=device)
C = torch.randn(intermediate_size, h, dtype=dtype, device=device)

# 矩阵乘法1
M1 = A @ B

# 矩阵乘法2
M2 = M1 @ C

# 输出张量的形状
print("M1 tensor shape:", M1.shape)  # 期望形状：(b * s, intermediate_size)
print("M2 tensor shape:", M2.shape)  # 期望形状：(b * s, h)

# 确认输入和输出张量的数据类型为 half
print("A tensor dtype:", A.dtype)
print("B tensor dtype:", B.dtype)
print("M1 tensor dtype:", M1.dtype)
print("C tensor dtype:", C.dtype)
print("M2 tensor dtype:", M2.dtype)

# 输出最大内存使用量
max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024
max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024 / 1024

print(f"Max memory allocated: {max_memory_allocated:.2f} MB")
print(f"Max memory reserved: {max_memory_reserved:.2f} MB")
