# 设置编译器和选项
NVCC = nvcc
ARCH = sm_90
CXXFLAGS = -DDEBUG -Xcompiler "-fopenmp -std=c++17" -std=c++17
INCLUDES = -I /home/zyhuang/cutlass/include -I /home/zyhuang/cutlass/tools/util/include
TARGET = test
SOURCES = matmul-v00.cu main.cu
OUTPUT = error.txt

# 默认目标
all: $(TARGET)

# 编译目标
$(TARGET): $(SOURCES)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) > $(OUTPUT) 2>&1

# 清理目标
clean:
	rm -f $(TARGET) $(OUTPUT)

# 执行目标
run: $(TARGET)
	./$(TARGET)
