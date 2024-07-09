import argparse
import os
from tqdm import tqdm

def generate_level1Z_order(M, N, t, execute_order):
    order = []
    for i in range(0, M, t):
        for j in range(N):
            for k in range(t):
                if i + k < M:
                    order.append((j, i + k, execute_order))
    return order

def generate_zigzag_order(M, N0, N1, t):
    M = int(M/128)
    order0 = generate_level1Z_order(M, N0, t, 0)
    order1 = generate_level1Z_order(M, N1, t, 1)
    new_order = []
    in_order_0 = True
    current_y_loc = 0
    while len(new_order) < M * (N0 + N1):
        if in_order_0:
            for i in range(t * N0):
                loc = i + current_y_loc * N0
                if(loc < M*N0):
                    new_order.append(order0[loc])
            in_order_0 = not in_order_0
        else:
            for j in range(t * N1):
                if(j + current_y_loc * N1<M*N1):
                    new_order.append(order1[j + current_y_loc * N1])
            in_order_0 = not in_order_0
            current_y_loc += t
    return new_order

def write_orders_to_file(f, description, orders):
    f.write(f'{description} [')
    f.write(', '.join([f'({x}, {y}, {z})' for x, y, z in orders]))
    f.write(']\n')

def main():
    parser = argparse.ArgumentParser(description='Generate zigzag order')
    parser.add_argument('--M', type=int, required=True, help='Number of rows')
    parser.add_argument('--file_path', type=str, required=True, help='Path to save the output file')
    args = parser.parse_args()
    
    M = args.M
    file_path = args.file_path
    N0 = 112  # 14336/128
    N1 = 32  # 4096/128
    
    with open(os.path.join(file_path, f'order_{M}.txt'), 'w') as f:
        for i in tqdm(range(6), desc="Generating orders for different t values"):
            t = 2 ** i
            order = generate_zigzag_order(M, N0, N1, t)
            write_orders_to_file(f, f't = {t} Order:', order)

if __name__ == "__main__":
    main()