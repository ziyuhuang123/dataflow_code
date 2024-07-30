1. 首先编译hzy_throughput.cu：
nvcc hzy_throughput.cu -o hzy_throughput -arch=sm_90 -std=c++17 -I./Common

2. 然后运行run_experiments.sh文件，遍历所有数值：
bash run_experiments.sh

不过这里我假设参数范围是：
nbins_values=(256 512 1024 2048)
block_size_values=(128 512)
cluster_size_values=(1 2 4 8 16)
array_size=2000000

3. 运行draw.py文件，利用上一步输出的results.csv来绘图：
python draw.py

得到throughput_comparison.png