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


我好像可以做kernel间L2数据流，然后选择A40，大M。然后不使用cutlass的swizzle。因为我们做的是kernel间。。。有点tricky，但是好像也可以。说不定再试试看3050那种显存很垃圾的显卡？可能更bound


缺点：
4. 本文对片上通信的操作基本没有做优化，只是用了GEMM的常规blocking操作而已。
5. 真的需要单独的CNN架构吗。。。GPU也可以吧。。只是想证明在不同配置上都能够达到最小访存量的话，其实GPU-sim也足够了？
6. 我看L2 throughput还没有拉满。。每次是循环到需要值的时候，才从DRAM读值。能不能一开始就制定好哪些值需要，然后一直连续不停的最大功率从DRAM读到L2（可以。。但是目前L2命中率都那么高了，明显不是bound。。。）
7. 本文。。也没有如其所说，对给定的硬件配置，选择合理的参数啊？甚至连最优tile size都没讨论。。


现在想想看两个GEMM的数据流，能不能用红蓝卵石方法得到通信下界？（L2上）


结论：没有想明白L2上最优memory access的理论值。。可以抽象为：有P个处理器连接一个高速缓存（L2），再连接一个低速缓存（DRAM），当下的模型都没有这一层L2。


明天的工作计划：
看论文
leanAttention，对比其与flash的区别
graphneIR，找到我的计算方法的表达



0607
给学长汇报：
1. 我的数据流设计思路
**针对decoding：统一所有kernel的block size，总体只有一个kernel。（可以先做？因为比较简单）

问：可能存在死锁？


**针对training：attention（用L1数据流）和projection融（L2数据流），FFN内部融（用flash和L2数据流）（这里没法用L1数据流）。描述两个GEMM的融合方案，有哪些参数可调？-->projection（不改变原尺寸，从(batch_size * seq_len) x d_model开始和结束。参考尺寸：8192*4096）
--->多个head的concatenate是按列来拼的。


问：仔细想一想GEMM1的计算顺序

答：
14336*4096*2/1024/1024=112MB，太大了。这是GEMM1的权重。那么我倾向于每次GEMM0计算都atomic加到全局内存的GEMM1的第一列上去，然后GEMM1的后几列等GEMM0都算完之后再单独计算。或者等GEMM0的对应一行都算完之后，再对GEMM1的一行进行计算（一次算完一整行吧）（不过我不看好这种。。因为GEMM1的权重太大了。。。）

一个想法：如果M比较大，横着Z好像不错，那如果M比较小，两个GEMM是不是就相当于N很大？纵向的N先向下，然后一路向右，N的高度较小，这样中间结果还能被利用，如何？（去看明白cutlass到底是怎么Z的）


下一步思考：从L2角度，attention-projection块和后面的FFN块怎么融？这样，我们仅考虑两个kernel间进行融合，先不考虑多个kernel混杂。然后我们限定前一个kernel的一行，作为最小粒度，执行完前一个行之后，才可以安排后面kernel的对应一行。这种方式就可以融合projection和FFN-GEMM0了。具体怎么排，实验验证吧。



在block执行速度不一样，或者同一种block但是对缓存的命中率不一样的情况下，不能认为先发射就先执行。每个block都需要验证依赖项计算完成。但是调整计算顺序可以最小化等待开销。



2. 简要阐述两篇paper看到了啥（文字即可）
3. 下周工作计划：尽快写代码（以及，冷老师希望L2数据流之类的先不要写模拟器找理论最优什么的，还是实测，要在GPU上拿到提升）
4. 问：要看flash attention吗？是不是大致搞明白怎么回事即可？全看懂好困难啊。。。（还是要硬啃的）
5. motivation：计算能力增长远快于内存速度增长（图来自冷老师的slide），memory wall优化是HPC的重要问题
6. CNN多了一个数据复用的维度，说不定优化空间更大？有时间可以看看：
“openai有一个开源的CLIP，有两种开源的backbone，一种实现是RESNET，另一种是ViT吧，大概是这样。。那个基于resnet的就是CNN”
“cnn based SD”
“在资源受限的边缘场景下还是会用CNN的，比如工厂的自动探测摄像头，就用resnet那种。大模型下面，那还是transformer占主导吧”
总之，也可以搞，但是好像意义不是特别大。


0608
1. 研究清楚decoding下的矩阵乘法GEMM/GEMV到底怎么写的

引用自《Flash Decoding++》：

> GEMV工作负载可以通过利用FastGEMV等先前设计中的CUDA Core来优化。对于解码阶段的Llama2-7B线性层，cuBLAS的张量核心实现仅达到在NVIDIA A100 GPU上使用FastGEMV的CUDA Core实现的82.15%的性能。另一方面，与张量核心实现相比，使用CUDA Core在batchsize=4的解码输入上进行投影仅达到49.75%的性能。

所以为了方便，我可以选batch=16，用cutlass/tensor core。

3. 测flash decoding并理解其细节，对比。


0611

1. 测试后发现对于M=16，N=14336（或乘10以确保wave足够多），K=4096，使用block_size=(128,256)并不比(64,....)要慢。实验在A40上测的很多，A100上稍微测了下。这保证了我可以使用大block，来和flash进行融合
2. 吐槽cusync的不足：使用多个stream，无法控制block的先后，对cutlass则必须使用cutlass::gemm::threadblock::CuSyncGemmHorizontalThreadblockSwizzle，这损害了L2的性能（虽然还没测其具体L2性能。。。）
3. 对于两个GEMM的执行顺序，可以算GEMM0的一行的一半，就立刻去算GEMM1的对应一行；可以尽快算完GEMM0的一行，然后就跑去算GEMM1的对应一行；可以算完GEMM0的好几行，然后再去算GEMM1的对应几行。
4. 修改cutlass的输入参数。现在我还需要输入D。判断，如果是consumer，就使用C和D，如果是producer就使用A和B。sync的方法暂时先不管。研究cutlass里面的block是二维的？执行顺序如何？