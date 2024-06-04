项目整体愿景：
使用数据流方法，在减少sync 开销的同时，充分利用各级缓存，减少memory traffic带来的开销。针对memory bound问题。原先的缓存命中率较低，经过数据流方法之后能够显著提高。现在需要找到合适的使用场景。

related work:

TileFlow: A Framework for Modeling Fusion Dataflow via Tree-based Analysis
Cusync: A Framework for Fine-Grained Synchronization of Dependent GPU Kernels

Welder: Scheduling Deep Learning Memory Access via Tile-graph

Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion

Souffle: Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions

| Framework | baseline                                                                            |
|-----------|-------------------------------------------------------------------------------------|
| TileFlow  |                     |
| Cusync |                |
| Welder    |                            |
| Chimera   |  |
| Souffle   | Bert等。。       |
| Optimus   |        |
| MASTRO   |        |

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

0604
关于L2数据流的一些思考：
1. 为了优先利用中间结果，GEMM0的一行算完之后立刻执行GEMM1的对应的一行
2. 由于FFN的形状，GEMM1的宽度较小，所以选择一次走完GEMM1的一行，而不切分。（如果切分，那么就无法享受中间结果了，就彻底丢到最后面再处理）
3. 由于依赖关系，GEMM1不存在hilbert式回环走法
4. GEMM1的值算完存回DRAM，然后再读上来再加。这可能造成一点额外开销。。
5. 但是考虑到残差结构，尽快算完GEMM0和GEMM1的一整行是有利的，这样输入的A还能尽可能留在L2，也就是说Z字形的纵向的高矮不要太高
6. 也应该用global memory access来衡量性能
7. 考虑到attention的残差结构，不应该在attention没算完，就把中间结果拿来算FFN。也就是说，attention和FFN的L2数据流是分开的。



paper 阅读思考：

卷积加速器中的通信下界

take away：
1. 数据流。。能加速，是不是也可以找数据额外流动带来的能量消耗？（这个肯定要做）
2. 我需要的数据流不仅是GEMM。。而应该是transformer架构下的多个kernel的综合设计。怎么理论上分析找到多个kernel的最小通信量？（可以用红蓝卵石方法吗？）优化空间有多大呢？
3. 加速器常用global SRAM buffer，然而GPU进行了切分。

```
话说为什么GPU的shared memory是每个SM内部的，而加速器往往是全局连起来的啊？

主要原因。。我猜是因为全局连起来的传输速度会很慢？比如SM0要访问存在SM107的shared memory的值。。
```

在局部切分情况下，最优通信是多少？（COSMA已经论证了）
如果全局连通，考虑上远距离访问的开销，数据怎么放置是最优的？（这个问题好像有点远）


我好像可以做kernel间L2数据流，然后选择A40，大M。然后不使用cutlass的swizzle。因为我们做的是kernel间。。。有点tricky，但是好像也可以。


缺点：
5. 本文对片上通信的操作基本没有做优化，只是用了GEMM的常规blocking操作而已。
7. 真的需要单独的CNN架构吗。。。GPU也可以吧。。只是想证明在不同配置上都能够达到最小访存量的话，其实GPU-sim也足够了？
8. 我看L2 throughput还没有拉满。。每次是循环到需要值的时候，才从DRAM读值。能不能一开始就制定好哪些值需要，然后一直连续不停的最大功率从DRAM读到L2（可以。。但是目前L2命中率都那么高了，明显不是bound。。。）
9. 本文。。也没有如其所说，对给定的硬件配置，选择合理的参数啊？甚至连最优tile size都没讨论。。


现在想想看两个GEMM的数据流，能不能用红蓝卵石方法得到通信下界？（L2上）

明天的工作计划：
看论文
leanAttention，对比其与flash的区别
graphneIR，找到我的计算方法的表达