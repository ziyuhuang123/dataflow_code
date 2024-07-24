import matplotlib.pyplot as plt
import numpy as np

# 数据
M_8B_FFN0 = np.array([256, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
M_8B_FFN1 = np.array([256, 1024, 2048, 4096, 16384, 32768, 65536, 131072])
M_70B_FFN0 = np.array([256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
M_70B_FFN1 = np.array([256, 1024, 2048, 4096, 16384, 32768, 65536, 131072])

# Lamma3-8B-A100-FFN0
global_access_swizzle1_8B_FFN0 = [1, 0.808572125, 0.476707889, 0.198841079, 0.05737762, 0.04624489, 0.042480771, 0.040716162]
global_access_swizzle2_8B_FFN0 = [1, 0.789566258, 0.449623602, 0.195938289, 0.100342201, 0.086191193, 0.080278623, 0.077591412]
global_access_swizzle128_8B_FFN0 = [0.952463749, 0.636047576, 0.38354402, 0.289433884, 0.242445318, 0.21994521, 0.20872442, 0.204468088]

# Lamma3-8B-A100-FFN1
global_access_swizzle1_8B_FFN1 = [0.508079, 0.337192, 0.224044, 0.103244, 0.023793, 0.021322, 0.02025, 0.019674]
global_access_swizzle2_8B_FFN1 = [0.508364, 0.34571, 0.211982, 0.096559, 0.042788, 0.040482, 0.039228, 0.038545]
global_access_swizzle128_8B_FFN1 = [0.508283, 0.309141, 0.197589, 0.136948, 0.108321, 0.101485, 0.097813, 0.096102]

# Lamma3-70B-A100-FFN0
global_access_swizzle1_70B_FFN0 = [0.707676422, 0.647707399, 0.354718281, 0.081237046, 0.03487307, 0.027090612, 0.023271622, 0.021551221, 0.020554642]
global_access_swizzle2_70B_FFN0 = [0.95606726, 0.645422663, 0.342959664, 0.09331946, 0.063648004, 0.050855459, 0.044286083, 0.041097676, 0.039583744]
global_access_swizzle128_70B_FFN0 = [0.901343643, 0.534568157, 0.302808289, 0.204708736, 0.152737102, 0.122910007, 0.10459156, 0.090393866, 0.078553787]

# Lamma3-70B-A100-FFN1
global_access_swizzle1_70B_FFN1 = [0.432574, 0.277791, 0.170537, 0.039577, 0.012881, 0.010806, 0.009828, 0.009364]
global_access_swizzle2_70B_FFN1 = [0.419646, 0.312724, 0.165977, 0.051865, 0.022727, 0.018636, 0.016241, 0.014111]
global_access_swizzle128_70B_FFN1 = [0.389426, 0.275394, 0.143693, 0.103452, 0.04116, 0.043775, 0.028638, 0.017356]

# 画图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Global Access: Idea/Real vs M')

# Lamma3-8B-A100-FFN0
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle1_8B_FFN0, marker='o', label='swizzle=1')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle2_8B_FFN0, marker='o', label='swizzle=2')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle128_8B_FFN0, marker='o', label='swizzle=128')
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_xlabel('M')
axs[0, 0].set_ylabel('Global Access: Idea/Real')
axs[0, 0].set_title('Lamma3-8B-A100-FFN0')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Lamma3-8B-A100-FFN1
axs[0, 1].plot(M_8B_FFN1, global_access_swizzle1_8B_FFN1, marker='o', label='swizzle=1')
axs[0, 1].plot(M_8B_FFN1, global_access_swizzle2_8B_FFN1, marker='o', label='swizzle=2')
axs[0, 1].plot(M_8B_FFN1, global_access_swizzle128_8B_FFN1, marker='o', label='swizzle=128')
axs[0, 1].set_xscale('log', base=2)
axs[0, 1].set_xlabel('M')
axs[0, 1].set_ylabel('Global Access: Idea/Real')
axs[0, 1].set_title('Lamma3-8B-A100-FFN1')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Lamma3-70B-A100-FFN0
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle1_70B_FFN0, marker='o', label='swizzle=1')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle2_70B_FFN0, marker='o', label='swizzle=2')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle128_70B_FFN0, marker='o', label='swizzle=128')
axs[1, 0].set_xscale('log', base=2)
axs[1, 0].set_xlabel('M')
axs[1, 0].set_ylabel('Global Access: Idea/Real')
axs[1, 0].set_title('Lamma3-70B-A100-FFN0')
axs[1, 0].legend()
axs[1, 0].grid(True)
# Lamma3-70B-A100-FFN1
axs[1, 1].plot(M_70B_FFN1, global_access_swizzle1_70B_FFN1, marker='o', label='swizzle=1')
axs[1, 1].plot(M_70B_FFN1, global_access_swizzle2_70B_FFN1, marker='o', label='swizzle=2')
axs[1, 1].plot(M_70B_FFN1, global_access_swizzle128_70B_FFN1, marker='o', label='swizzle=128')
axs[1, 1].set_xscale('log', base=2)
axs[1, 1].set_xlabel('M')
axs[1, 1].set_ylabel('Global Access: Idea/Real')
axs[1, 1].set_title('Lamma3-70B-A100-FFN1')
axs[1, 1].legend()
axs[1, 1].grid(True)
plt.tight_layout()
plt.savefig('global_access_plot_4figs.png')
plt.show()
