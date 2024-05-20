import torch
import torch.nn as nn

# 设置参数
b = 2
s = 16384
h = 1024
dtype = torch.float

# 检查是否有可用的 GPU 并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化稀疏矩阵
A = torch.randn(b * s, h, dtype=dtype, device=device)
B = torch.randn(h, 4 * h, dtype=dtype, device=device)
C = torch.randn(4 * h, h, dtype=dtype, device=device)

# 稀疏化矩阵B和C
B[B.abs() < 0.9] = 0
C[C.abs() < 0.9] = 0
B = B.to_sparse()
C = C.to_sparse()

# 矩阵乘法1
M1 = torch.sparse.mm(A, B)

# 矩阵乘法2
M2 = torch.sparse.mm(M1, C)

# 输出张量的形状
print("M1 tensor shape:", M1.shape)  # 期望形状：(b * s, 4 * h)
print("M2 tensor shape:", M2.shape)  # 期望形状：(b * s,  h)

# 确认输入和输出张量的数据类型
print("A tensor dtype:", A.dtype)
print("B tensor dtype:", B.dtype)
print("M1 tensor dtype:", M1.dtype)
print("C tensor dtype:", C.dtype)
print("M2 tensor dtype:", M2.dtype)
