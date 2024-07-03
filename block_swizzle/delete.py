def generate_zigzag_order(M, N, t):
    order = []
    down = True
    
    for i in range(0, M, t):
        for j in range(0, N):
            if down:
                for k in range(t):
                    if i + k < M:
                        order.append((i + k, j))
            else:
                for k in range(t-1, -1, -1):
                    if i + k < M:
                        order.append((i + k, j))
        down = not down
    
    return order

# 示例
M = 4
N = 5
t = 2
order = generate_zigzag_order(M, N, t)
print(order)
