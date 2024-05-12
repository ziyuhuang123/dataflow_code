import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QGridLayout, QLabel, \
    QLineEdit, QHBoxLayout
from collections import deque


class MatrixBlockApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.history = []  # To store the history of operations
        self.global_memory_access = 0.0  # Total global memory access in MB

    def initUI(self):
        self.setWindowTitle('Matrix Block Computation Simulation')
        self.setGeometry(100, 100, 1200, 600)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        control_layout = QVBoxLayout()
        layout.addLayout(control_layout)

        self.m_input = QLineEdit(self)
        self.n_input = QLineEdit(self)
        self.k_input = QLineEdit(self)
        self.cache_size_input = QLineEdit(self)
        self.start_button = QPushButton('Start Simulation', self)
        self.undo_button = QPushButton('Undo Last Step', self)
        self.start_button.clicked.connect(self.start_simulation)
        self.undo_button.clicked.connect(self.undo_last_step)
        self.undo_button.setEnabled(False)

        control_layout.addWidget(QLabel('M: (Number of rows in Matrix A and C)'))
        control_layout.addWidget(self.m_input)
        control_layout.addWidget(QLabel('N: (Number of columns in Matrix B and C)'))
        control_layout.addWidget(self.n_input)
        control_layout.addWidget(QLabel('K: (Number of columns in Matrix A and rows in Matrix B)'))
        control_layout.addWidget(self.k_input)
        control_layout.addWidget(QLabel('Cache Size: (Maximum number of blocks in cache)'))
        control_layout.addWidget(self.cache_size_input)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.undo_button)

        self.memory_access_label = QLabel('Global Memory Access: 0.0 MB', self)
        control_layout.addWidget(self.memory_access_label)

        self.cache_label = QLabel('Cache: None', self)
        control_layout.addWidget(self.cache_label)

        self.grid_layout = QGridLayout()
        layout.addLayout(self.grid_layout)

    def start_simulation(self):
        try:
            M = int(self.m_input.text() or 512)
            N = int(self.n_input.text() or 512)
            K = int(self.k_input.text() or 128)
            cache_size = int(self.cache_size_input.text() or 6)

            self.setup_simulation(M, N, K, cache_size)
        except ValueError as e:
            self.cache_label.setText(f"Input Error: {str(e)}")

    def setup_simulation(self, M, N, K, cache_size):
        self.M, self.N, self.K = M, N, K
        self.cache_capacity = cache_size
        self.block_size = 128
        self.grid_size_x = self.M // self.block_size
        self.grid_size_y = self.N // self.block_size
        self.cache = deque(maxlen=self.cache_capacity)
        self.buttons = {}

        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                btn = QPushButton(f'Block {i + 1},{j + 1}', self)
                btn.clicked.connect(self.make_compute_block(i, j))
                self.grid_layout.addWidget(btn, i, j)
                self.buttons[(i, j)] = btn

        self.A = np.random.rand(self.M, self.K)
        self.B = np.random.rand(self.K, self.N)
        self.history.clear()
        self.undo_button.setEnabled(False)
        self.global_memory_access = 0.0
        self.update_memory_access_label()

    def update_cache_and_memory_access(self, item, type):
        entry = f'{type}: {item}'
        print("entry", entry)
        print("self.cache", self.cache)
        # Check if the block is not already in cache
        if entry not in self.cache:
            print("enter entry not in self.cache:")

            # If the cache is full and an item needs to be removed, consider it as a memory write access
            if len(self.cache) == self.cache_capacity:
                oldest_entry = self.cache[-1]
                # Check if the oldest entry is a 'C' block, which needs to be written back to memory
                if 'C:' in oldest_entry:
                    self.update_memory_access(self.block_size, self.block_size, 'add')
                    print("finish C") # 下面这几行是不是应该放到什么else里面去？
            self.cache.appendleft(entry)
            if type != 'C':
                # Only add to memory access if the block was not in the cache
                self.update_memory_access(self.block_size, self.block_size, 'add')
                print("finish A/B")

    def make_compute_block(self, i, j):
        def compute_block():
            self.buttons[(i, j)].setStyleSheet("background-color: red")
            # Handle memory access for all blocks of A needed for this computation
            for k in range(1, self.K // self.block_size + 1):  # Assuming K is perfectly divisible by block_size
                self.update_cache_and_memory_access((i + 1, k), 'A')
                print(f"(i + 1, {k}), 'A'")

            # Handle memory access for all blocks of B needed for this computation
            for k in range(1, self.K // self.block_size + 1):
                self.update_cache_and_memory_access((k, j + 1), 'B')
                print(f"({k}, j + 1), 'B'")

            # Check if C is already in cache or needs to be added
            self.update_cache_and_memory_access((i + 1, j + 1), 'C')
            print(f"(i + 1, j + 1), 'C'")
            self.update_cache_label()
            self.record_state(i, j)

        return compute_block

    def update_memory_access(self, rows, cols, action='add'):
        # Calculate memory accessed for a block of given rows and cols in MB assuming float16 format
        # block_memory = rows * cols * 2 / 1024 / 1024  # 2 bytes per float16
        block_memory = 1
        if action == 'add':
            self.global_memory_access += block_memory
        elif action == 'subtract':
            self.global_memory_access -= block_memory
        print("self.global_memory_access=", self.global_memory_access)
        self.update_memory_access_label()

    def update_memory_access_label(self):
        # Display memory access with more decimal points for precision
        self.memory_access_label.setText(f'Global Memory Access: {self.global_memory_access:.5f} MB')

    def update_cache(self, item, type):
        entry = f'{type}: {item}'
        if entry not in self.cache:
            if len(self.cache) == self.cache_capacity:
                self.update_memory_access(self.block_size, self.block_size)  # Assuming a block is being replaced
            self.cache.appendleft(entry)

    def update_cache_label(self):
        formatted_cache = ', '.join(self.cache)
        self.cache_label.setText(f"Cache updated with: {formatted_cache}")

    def record_state(self, i, j):
        state = {
            'cache': list(self.cache),
            'buttons': {key: btn.styleSheet() for key, btn in self.buttons.items()},
            'memory_access': self.global_memory_access
        }
        self.history.append(state)
        self.undo_button.setEnabled(True)

    def undo_last_step(self):
        if self.history:
            self.history.pop()
            if self.history:
                last_state = self.history[-1]
                self.cache = deque(last_state['cache'], maxlen=self.cache_capacity)
                self.global_memory_access = last_state['memory_access']
                self.update_memory_access_label()
                for key, style in last_state['buttons'].items():
                    self.buttons[key].setStyleSheet(style)
                self.update_cache_label()
            else:
                self.reset_simulation()
        self.undo_button.setEnabled(len(self.history) > 0)

    def reset_simulation(self):
        for btn in self.buttons.values():
            btn.setStyleSheet("")
        self.cache.clear()
        self.update_cache_label()
        self.global_memory_access = 0.0
        self.update_memory_access_label()


if __name__ == '__main__':
    app = QApplication([])
    ex = MatrixBlockApp()
    ex.show()
    app.exec_()
