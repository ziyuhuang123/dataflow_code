0512
1. 今天写了GUI，能够设定矩阵乘法的MNK尺寸，cache的大小。每次点击方框能够计算当前cache内的值和截至目前的global memory access。并且有undo按钮回撤到上一步。作用是帮助理解为什么不同Z字形的memory access不一样，想获得某种intuition。
2. 打算明天用暴力搜索的方式，对3*3或者4*4来搜索，找到最优的所有路径。目标是最小化global memory access。总共有组合数A_4_4种可能，当然限定初始必须从0,0开始。
3. 据此的intuition来思考，怎么描述所有路径？真的必须暴力搜索吗？目前还是没有思路。。


我的互动的图需要加上每一步的箭头。更容易看出路径

0520
今天测了FFN的性能。主要关注的是是否memory bound以及L2命中率。。。针对lamma3-8B和lamma3-70B在A100，A40,4090和本机的3050上测了。结果大多都是compute bound，L2命中率都在80%以上。在本机的3050上偶尔能memory bound。总之就是性能很不如预期。。。

那么怎么寻找memory bound的场景呢？1. 稀疏矩阵乘法？2. decoding（此时attention和FFN好像都是memory bound）