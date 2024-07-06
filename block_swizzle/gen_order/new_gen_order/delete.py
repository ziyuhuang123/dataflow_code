def read_first_lines(file_path, num_lines=20):
    try:
        with open(file_path, 'r') as file:
            for _ in range(num_lines):
                line = file.readline()
                if not line:  # 如果到达文件末尾
                    break
                print(line.strip())
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
file_path = 'orders.txt'
read_first_lines(file_path)
