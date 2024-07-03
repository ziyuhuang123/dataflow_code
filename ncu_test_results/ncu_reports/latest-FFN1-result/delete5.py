import matplotlib.pyplot as plt
import numpy as np

# 数据
M_8B_FFN0 = np.array([256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
M_8B_FFN1 = np.array([256, 1024, 2048, 16384, 32768, 65536, 131072])
M_70B_FFN0 = np.array([256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
M_70B_FFN1 = np.array([256, 1024, 2048, 16384, 32768, 65536, 131072])

# Lamma3-8B-A40-FFN0
global_access_swizzle1_8B_FFN0 = [0.933179113, 0.922142786, 0.862878406, 0.692939063, 0.379669233, 0.348991841, 0.349979631, 0.355823028, 0.352974256]
global_access_swizzle2_8B_FFN0 = [0.933464644, 0.914339996, 0.856084581, 0.684792642, 0.66994722, 0.670326954, 0.662481377, 0.667014623, 0.668149746]
global_access_swizzle4_8B_FFN0 = [0.919959017, 0.90084482, 0.843876666, 0.8124199, 0.813402467, 0.805502373, 0.792202241, 0.758733371, 0.718541054]
global_access_swizzle8_8B_FFN0 = [0.892662231, 0.874282253, 0.856357769, 0.859940687, 0.861220882, 0.859840878, 0.806597073, 0.730891673, 0.689375935]
global_access_swizzle16_8B_FFN0 = [0.892690784, 0.874390625, 0.856343971, 0.859940687, 0.861220882, 0.853132421, 0.800069184, 0.731811368, 0.68615858]
global_access_swizzle32_8B_FFN0 = [0.892719337, 0.874276549, 0.856211517, 0.859940687, 0.861220882, 0.856822072, 0.787850315, 0.736660673, 0.681645925]
global_access_swizzle128_8B_FFN0 = [0.893547378, 0.874014175, 0.85624463, 0.859940687, 0.860547384, 0.855480381, 0.822163576, 0.753215196, 0.684696145]

# Lamma3-8B-A40-FFN1
global_access_swizzle1_8B_FFN1 = [0.949671783, 0.925939712, 0.868143354, 0.280857249, 0.258630624, 0.25000259, 0.244511777]
global_access_swizzle2_8B_FFN1 = [0.950773291, 0.921448437, 0.867216481, 0.681840118, 0.682346687, 0.681584526, 0.649257107]
global_access_swizzle128_8B_FFN1 = [0.949787731, 0.889934466, 0.872032313, 0.871815082, 0.822174681, 0.76808694, 0.724391498]

# Lamma3-70B-A40-FFN0
global_access_swizzle1_70B_FFN0 = [0.926514708, 0.919663113, 0.738507471, 0.645140765, 0.293643663, 0.287971494, 0.290204046, 0.290774802, 0.289923699]
global_access_swizzle2_70B_FFN0 = [0.926550293, 0.912550387, 0.778427441, 0.673927099, 0.661384834, 0.662968712, 0.638621503, 0.628694347, 0.622320589]
global_access_swizzle4_70B_FFN0 = [0.906601419, 0.894057299, 0.821100512, 0.809392201, 0.808380505, 0.770230299, 0.722751062, 0.673449969, 0.646167449]
global_access_swizzle8_70B_FFN0 = [0.893527541, 0.86702894, 0.857579106, 0.859852951, 0.85541912, 0.806123607, 0.739159458, 0.68542095, 0.653025506]
global_access_swizzle16_70B_FFN0 = [0.893328265, 0.86702894, 0.857579106, 0.860191614, 0.853907165, 0.800768895, 0.745380452, 0.674471882, 0.664615414]
global_access_swizzle32_70B_FFN0 = [0.893477722, 0.86702894, 0.857579106, 0.859514288, 0.851219244, 0.807796955, 0.724087115, 0.682668042, 0.656048054]
global_access_swizzle128_70B_FFN0 = [0.893093405, 0.865606394, 0.857579106, 0.859852951, 0.847691348, 0.798844545, 0.734316267, 0.678851509, 0.657788625]

# Lamma3-70B-A40-FFN1
global_access_swizzle1_70B_FFN1 = [0.92215466, 0.919180899, 0.859521128, 0.190050973, 0.195386341, 0.183050602, 0.181954276]
global_access_swizzle2_70B_FFN1 = [0.921122111, 0.916318731, 0.862290217, 0.676715661, 0.674751112, 0.652966742, 0.482021339]
global_access_swizzle128_70B_FFN1 = [0.900901347, 0.876248381, 0.86298249, 0.799054952, 0.750789441, 0.721820384, 0.66176391]

# 画图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Global Access: Idea/Real vs M')

# Lamma3-8B-A40-FFN0
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle1_8B_FFN0, marker='o', label='swizzle=1')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle2_8B_FFN0, marker='o', label='swizzle=2')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle4_8B_FFN0, marker='o', label='swizzle=4')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle8_8B_FFN0, marker='o', label='swizzle=8')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle16_8B_FFN0, marker='o', label='swizzle=16')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle32_8B_FFN0, marker='o', label='swizzle=32')
axs[0, 0].plot(M_8B_FFN0, global_access_swizzle128_8B_FFN0, marker='o', label='swizzle=128')
axs[0, 0].set_xscale('log', base=2)
# 设置X轴和Y轴的标签、标题和图例
axs[0, 0].set_xlabel('M')
axs[0, 0].set_ylabel('Global Access: Idea/Real')
axs[0, 0].set_title('Lamma3-8B-A40-FFN0')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Lamma3-8B-A40-FFN1
axs[0, 1].plot(M_8B_FFN1, global_access_swizzle1_8B_FFN1, marker='o', label='swizzle=1')
axs[0, 1].plot(M_8B_FFN1, global_access_swizzle2_8B_FFN1, marker='o', label='swizzle=2')
axs[0, 1].plot(M_8B_FFN1, global_access_swizzle128_8B_FFN1, marker='o', label='swizzle=128')
axs[0, 1].set_xscale('log', base=2)
axs[0, 1].set_xlabel('M')
axs[0, 1].set_ylabel('Global Access: Idea/Real')
axs[0, 1].set_title('Lamma3-8B-A40-FFN1')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Lamma3-70B-A40-FFN0
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle1_70B_FFN0, marker='o', label='swizzle=1')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle2_70B_FFN0, marker='o', label='swizzle=2')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle4_70B_FFN0, marker='o', label='swizzle=4')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle8_70B_FFN0, marker='o', label='swizzle=8')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle16_70B_FFN0, marker='o', label='swizzle=16')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle32_70B_FFN0, marker='o', label='swizzle=32')
axs[1, 0].plot(M_70B_FFN0, global_access_swizzle128_70B_FFN0, marker='o', label='swizzle=128')
axs[1, 0].set_xscale('log', base=2)
axs[1, 0].set_xlabel('M')
axs[1, 0].set_ylabel('Global Access: Idea/Real')
axs[1, 0].set_title('Lamma3-70B-A40-FFN0')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Lamma3-70B-A40-FFN1
axs[1, 1].plot(M_70B_FFN1, global_access_swizzle1_70B_FFN1, marker='o', label='swizzle=1')
axs[1, 1].plot(M_70B_FFN1, global_access_swizzle2_70B_FFN1, marker='o', label='swizzle=2')
axs[1, 1].plot(M_70B_FFN1, global_access_swizzle128_70B_FFN1, marker='o', label='swizzle=128')
axs[1, 1].set_xscale('log', base=2)
axs[1, 1].set_xlabel('M')
axs[1, 1].set_ylabel('Global Access: Idea/Real')
axs[1, 1].set_title('Lamma3-70B-A40-FFN1')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('global_access_plot_4figs_updated.png')
plt.show()

