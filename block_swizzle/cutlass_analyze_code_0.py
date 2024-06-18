x_max = 10
log_tile = 1

for x_value in range(0, x_max):
    transformed_x = x_value >> log_tile
    print("x transformation", x_value, "-->", transformed_x)

print("=================================")

y_max = 3

for new_x in range(0, x_max):
    for y_value in range(0, y_max):
        transformed_y_1 = (new_x & ((1 << log_tile) - 1))
        print("new_x & 1:",new_x, "-->", transformed_y_1)
        transformed_y_0 = (y_value << log_tile)
        print("y_value << log_tile:", y_value, "-->", transformed_y_0)
        transformed_y = transformed_y_0 + transformed_y_1
        print("y transformation", "--->", transformed_y)
        print("")

print("=================================")
for x_value in range(0, x_max):
    transformed_x = x_value & ((1 << 2) - 1) # a & b ，那么结果不会比b大，就必然是[0, b)范围内的整数
    print("new_x & 1:",x_value, "-->", transformed_x)

# 当x增长，对transformed_x是0 0 1 1这样以log_tile为延迟的阶梯状增长。x的增长对于transformed_y是0 1 0 1 0 1这样反复震荡的变换，所以现在我们先保持y不变，那么就已经实现了先向下，再右上，再向下这样的倒N型变换。
# 当y增长，对于transformed_x没有影响，对于transformed_y则会额外增加y << log_tile就是扩增，这样使得后续的N和之前的N不会重叠。
# 需要多出来一些初始block，是因为N型的表示空间需要是log_tile


# x_useful_max* (1<<log_tile)
# x:5--->10
# transformed_x = block_idx_x >> log_tile
# 10--->5
# 这两步能保证还原回来吗？可以。
#
# 然后证明：new_y_max>=y_useful_max
# new_y_max = 1<<log_tile-1 + y_max <<log_tile
#           = 1<<log_tile -1 + y_max * (1<<log_tile)
#           = 1<<log_tile -1 + 向下取整((y_useful_max+(1<<log_tile-1))/(1<<log_tile))*(1<<log_tile)
#           = 1<<log_tile-1 + 向下取整[(y_useful_max-1)/(1<<log_tile)+1]*(1<<log_tile) (提取出一个1来)
#
# 若log_tile>=1，则：
#     new_y_max >= 1<<log_tile -1 + (y_useful_max-1)/(1<<log_tile) * (1<<log_tile)
#             = 1<<log_tile -1 + y_useful_max -1 >= y_useful_max
# 若log_tile=0，则：
#     new_y_max = 1<<log_tile -1 + y_max<<log_tile
#               = 1-1 + y_max = 向下取整(y_useful_max+ (1<<log_tile -1)/(1<<log_tile))
#               = y_useful_max
# 综上，得证new_y_max>=y_useful_max
#
# 不过这个方式显然无法表示复杂的嵌套Z字形。我认为不如生成一个序列，然后放到local memory，每个block取自己需要执行的index。
#
# 所以我要：把原先的二维index转换为一维，然后因为目前就是decoding，就线性计算就好。之后对training可以搞多维Z字嵌套。