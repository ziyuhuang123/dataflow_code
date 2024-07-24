import matplotlib.pyplot as plt
import numpy as np

# 数据
M_8B_FFN0 = np.array([256, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
M_8B_FFN1 = np.array([256, 1024, 2048, 4096, 16384, 32768, 65536, 131072])
M_70B_FFN0 = np.array([256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
M_70B_FFN1 = np.array([256, 1024, 2048, 4096, 16384, 32768, 65536, 131072])

# Lamma3-8B-A100-FFN1
global_access_swizzle1_8B_FFN1 = [0.945387027, 0.937581776, 0.916979729, 0.694202693, 0.466197845, 0.45150661, 0.445483016, 0.443502828]
global_access_swizzle2_8B_FFN1 = [0.932910386, 0.953270748, 0.913518367, 0.772880824, 0.703556262, 0.687053283, 0.668822165, 0.634303094]
global_access_swizzle128_8B_FFN1 = [0.900514141, 0.93635963, 0.893442471, 0.899242671, 0.842639376, 0.87372717, 0.816994866, 0.704653093]

# Lamma3-8B-A100-FFN0
global_access_swizzle1_8B_FFN0 = [1, 0.979043833, 0.941556538, 0.853152004, 0.535920957, 0.500938403, 0.499953688, 0.500967738]
global_access_swizzle2_8B_FFN0 = [1, 0.97640868, 0.934828939, 0.85043653, 0.746726085, 0.743449141, 0.745837494, 0.748198674]
global_access_swizzle4_8B_FFN0 = [1, 0.966250223, 0.924075268, 0.866729371, 0.865935374, 0.867555604, 0.87003476, 0.873528121]
global_access_swizzle8_8B_FFN0 = [0.982319038, 0.9495267, 0.914704093, 0.910037102, 0.912406792, 0.913843961, 0.915897362, 0.91809156]
global_access_swizzle16_8B_FFN0 = [0.981490997, 0.949720629, 0.914960724, 0.910730905, 0.911733293, 0.914179384, 0.915897362, 0.918258777]
global_access_swizzle32_8B_FFN0 = [0.981833635, 0.949327067, 0.914414348, 0.910950858, 0.911733293, 0.913843961, 0.916064744, 0.919763734]
global_access_swizzle128_8B_FFN0 = [0.981919294, 0.949349883, 0.914428146, 0.910523171, 0.911733293, 0.914179384, 0.915897362, 0.917589908]

# Lamma3-70B-A100-FFN0
global_access_swizzle1_70B_FFN0 = [0.856348476, 0.957811508, 0.922276988, 0.704406747, 0.500781564, 0.495884945, 0.494327826, 0.498014604, 0.496728512]
global_access_swizzle2_70B_FFN0 = [0.984019846, 0.95738759, 0.918147336, 0.746062266, 0.734630677, 0.738018356, 0.739994491, 0.742022422, 0.743743634]
global_access_swizzle4_70B_FFN0 = [0.977984653, 0.948394259, 0.909888032, 0.857143649, 0.854579145, 0.85690971, 0.856815586, 0.856664386, 0.856307487]
global_access_swizzle8_70B_FFN0 = [0.961679669, 0.93246602, 0.901628728, 0.899476493, 0.899937809, 0.89974741, 0.89681366, 0.885006832, 0.876110387]
global_access_swizzle16_70B_FFN0 = [0.962298846, 0.93246602, 0.901628728, 0.899476493, 0.899937809, 0.899914745, 0.898692483, 0.886362431, 0.874025871]
global_access_swizzle32_70B_FFN0 = [0.964220428, 0.93246602, 0.901628728, 0.899815156, 0.899937809, 0.899831078, 0.894642574, 0.888739943, 0.876110387]
global_access_swizzle128_70B_FFN0 = [0.96193588, 0.93246602, 0.901628728, 0.898460505, 0.899937809, 0.899831078, 0.896855411, 0.888739943, 0.876110387]

# Lamma3-70B-A100-FFN1
global_access_swizzle1_70B_FFN1 = [0.945387027, 0.937581776, 0.916979729, 0.694202693, 0.466197845, 0.45150661, 0.445483016, 0.443502828]
global_access_swizzle2_70B_FFN1 = [0.932910386, 0.953270748, 0.913518367, 0.772880824, 0.703556262, 0.687053283, 0.668822165, 0.634303094]
global_access_swizzle128_70B_FFN1 = [0.900514141, 0.93635963, 0.893442471, 0.899242671, 0.842639376, 0.87372717, 0.816994866, 0.704653093]

# 画图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Global Access: Idea/Real vs M')

# Lamma3-8B-A100-FFN1
axs[0, 0].plot(M_8B_FFN1, global_access_swizzle1_8B_FFN1, marker='o', label='swizzle=1')
axs[0, 0].plot(M_8B_FFN1, global_access_swizzle2_8B_FFN1, marker='o', label='swizzle=2')
axs[0, 0].plot(M_8B_FFN1, global_access_swizzle128_8B_FFN1, marker='o', label='swizzle=128')
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_xlabel('M')
axs[0, 0].set_ylabel('Global Access: Idea/Real')
axs[0, 0].set_title('Lamma3-8B-A100-FFN1')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Lamma3-8B-A100-FFN0
axs[0, 1].plot(M_8B_FFN0, global_access_swizzle1_8B_FFN0, marker='o', label='swizzle=1')
axs[0, 1].plot(M_8B_FFN0, global_access_swizzle2_8B_FFN0, marker='o', label='swizzle=2')
axs[0, 1].plot(M_8B_FFN0, global_access_swizzle4_8B_FFN0, marker='o', label='swizzle=4')
axs[0, 1].plot(M_8B_FFN0, global_access_swizzle8_8B_FFN0, marker='o', label='swizzle=8')
axs[0, 1].plot(M_8B_FFN0, global_access_swizzle16_8B_FFN0, marker='o', label='swizzle=16')
axs[0, 1].plot(M_8B_FFN0, global_access_swizzle32_8B_FFN0, marker='o', label='swizzle=32')
axs[0, 1].plot(M_8B_FFN0, global_access_swizzle128_8B_FFN0, marker='o', label='swizzle=128')
axs[0, 1].set_xscale('log', base=2)
axs[0, 1].set_xlabel('M')
axs[0, 1].set_ylabel('Global Access: Idea/Real')
axs[0, 1].set_title('Lamma3-8B-A100-FFN0')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Lamma3-70B-A100-FFN0
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
plt.savefig('global_access_plot_updated.png')
plt.show()
