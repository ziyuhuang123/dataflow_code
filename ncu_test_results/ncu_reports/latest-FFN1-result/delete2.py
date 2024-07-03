import matplotlib.pyplot as plt
import numpy as np

# 数据
M_8B_FFN0 = np.array([256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
M_8B_FFN1 = np.array([256, 1024, 4096, 16384, 32768, 65536, 131072])
M_70B_FFN0 = np.array([256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
M_70B_FFN1 = np.array([256, 1024, 2048, 16384, 32768, 65536, 131072])

# Lamma3-8B-A40-FFN0
global_access_swizzle1_8B_FFN0 = [0.844275326, 0.532034859, 0.279681067, 0.106100971, 0.043554559, 0.035838067, 0.03300299, 0.031834124, 0.030971707]
global_access_swizzle2_8B_FFN0 = [0.844837502, 0.508200701, 0.270043506, 0.103643033, 0.078840301, 0.06838107, 0.061675201, 0.059805468, 0.058661594]
global_access_swizzle128_8B_FFN0 = [0.772890448, 0.412660537, 0.270262903, 0.206488812, 0.168445944, 0.143423517, 0.110912139, 0.079043982, 0.061550537]

# Lamma3-8B-A40-FFN1
global_access_swizzle1_8B_FFN1 = [0.439814, 0.273721, 0.145013, 0.016455, 0.014687, 0.013884, 0.013468]
global_access_swizzle2_8B_FFN1 = [0.440976, 0.26641, 0.144293, 0.035713, 0.032986, 0.031517, 0.028136]
global_access_swizzle128_8B_FFN1 = [0.439936, 0.224359, 0.148115, 0.080157, 0.056018, 0.042279, 0.035266]

# Lamma3-70B-A40-FFN0
global_access_swizzle1_70B_FFN0 = [0.825552748, 0.491224793, 0.140442758, 0.068600172, 0.024901248, 0.019333044, 0.016690851, 0.015350494, 0.014655981]
global_access_swizzle2_70B_FFN0 = [0.825622493, 0.47005131, 0.161655467, 0.074206917, 0.050576629, 0.039983832, 0.032263877, 0.028916768, 0.027203862]
global_access_swizzle128_70B_FFN0 = [0.764868774, 0.365945987, 0.230764681, 0.15717962, 0.105891699, 0.065230233, 0.043380655, 0.033282694, 0.029938956]

# Lamma3-70B-A40-FFN1
global_access_swizzle1_70B_FFN1 = [0.409106, 0.245612, 0.117123, 0.008565, 0.007417, 0.006713, 0.006408]
global_access_swizzle2_70B_FFN1 = [0.408122, 0.241265, 0.118918, 0.020919, 0.017956, 0.015522, 0.010046]
global_access_swizzle128_70B_FFN1 = [0.389761, 0.193354, 0.119376, 0.032819, 0.023181, 0.019217, 0.015222]

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 第一个子图
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle1_8B_FFN0, marker='o', label='swizzle=1')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle2_8B_FFN0, marker='o', label='swizzle=2')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle128_8B_FFN0, marker='o', label='swizzle=128')
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_xlabel('M (log scale)')
axs[0, 0].set_ylabel('global access:idea/real')
axs[0, 0].set_title('Lamma3-8B-A40-FFN0')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 第二个子图
axs[0, 1].plot(M_8B_FFN1, global_access_swizzle1_8B_FFN1, marker='o', label='swizzle=1')
axs[0, 1].plot(M_8B_FFN1, global_access_swizzle2_8B_FFN1, marker='o', label='swizzle=2')
axs[0, 1].plot(M_8B_FFN1, global_access_swizzle128_8B_FFN1, marker='o', label='swizzle=128')
axs[0, 1].set_xscale('log', base=2)
axs[0, 1].set_xlabel('M (log scale)')
axs[0, 1].set_ylabel('global access:idea/real')
axs[0, 1].set_title('Lamma3-8B-A40-FFN1')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 第三个子图
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle1_70B_FFN0, marker='o', label='swizzle=1')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle2_70B_FFN0, marker='o', label='swizzle=2')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle128_70B_FFN0, marker='o', label='swizzle=128')
axs[1, 0].set_xscale('log', base=2)
axs[1, 0].set_xlabel('M (log scale)')
axs[1, 0].set_ylabel('global access:idea/real')
axs[1, 0].set_title('Lamma3-70B-A40-FFN0')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 第四个子图
axs[1, 1].plot(M_70B_FFN1, global_access_swizzle1_70B_FFN1, marker='o', label='swizzle=1')
axs[1, 1].plot(M_70B_FFN1, global_access_swizzle2_70B_FFN1, marker='o', label='swizzle=2')
axs[1, 1].plot(M_70B_FFN1, global_access_swizzle128_70B_FFN1, marker='o', label='swizzle=128')
axs[1, 1].set_xscale('log', base=2)
axs[1, 1].set_xlabel('M (log scale)')
axs[1, 1].set_ylabel('global access:idea/real')
axs[1, 1].set_title('Lamma3-70B-A40-FFN1')
axs[1, 1].legend()
axs[1, 1].grid(True)

# 调整子图布局
plt.tight_layout()

# 保存图像到文件
output_file = 'global_access_subplots.png'
plt.savefig(output_file)
plt.show()
