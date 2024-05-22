项目整体愿景：
使用数据流方法，在减少sync 开销的同时，充分利用各级缓存，减少memory traffic带来的开销。针对memory bound问题。原先的缓存命中率较低，经过数据流方法之后能够显著提高。现在需要找到合适的使用场景。

related work:
TileFlow: A Framework for Modeling Fusion Dataflow via Tree-based Analysis
A Framework for Fine-Grained Synchronization of Dependent GPU Kernels
Welder: Scheduling Deep Learning Memory Access via Tile-graph
Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion



0512
1. 今天写了GUI，能够设定矩阵乘法的MNK尺寸，cache的大小。每次点击方框能够计算当前cache内的值和截至目前的global memory access。并且有undo按钮回撤到上一步。作用是帮助理解为什么不同Z字形的memory access不一样，想获得某种intuition。
2. 打算明天用暴力搜索的方式，对3*3或者4*4来搜索，找到最优的所有路径。目标是最小化global memory access。总共有组合数A_4_4种可能，当然限定初始必须从0,0开始。
3. 据此的intuition来思考，怎么描述所有路径？真的必须暴力搜索吗？目前还是没有思路。。


我的互动的图需要加上每一步的箭头。更容易看出路径

0520
今天测了FFN的性能。主要关注的是是否memory bound以及L2命中率。。。针对lamma3-8B和lamma3-70B在A100，A40,4090和本机的3050上测了。结果大多都是compute bound，L2命中率都在80%以上。在本机的3050上偶尔能memory bound。总之就是性能很不如预期。。。

那么怎么寻找memory bound的场景呢？1. 稀疏矩阵乘法？2. decoding（此时attention和FFN好像都是memory bound）

0521
1. GUI_cache.py是什么？
之前想探究最优的GEMM路径，为了寻找intuition，做了这个GUI，能够手动去计算每次的计算路径，然后显示出当前步的总的global memory access，并且能够回退到上一步。
2. attention的哪部分是memory bound？
attention一般有两个GEMM，我们先不考虑element-wise操作，那就是第一个GEMM比较memory bound。原因是，K维度很小，计算复用很少。而L2命中率一般是第二个GEMM更低，因为Q*K的中间结果很大，难以存到缓存上。

下一步工作计划：
测试不同参数下的memory bound和L2命中率情况。寻找合适场景进行实验。

0522
