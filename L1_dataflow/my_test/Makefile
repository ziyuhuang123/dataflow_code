# Makefile for cublas_test

# Compiler and flags
NVCC = nvcc
CFLAGS = -lcublas -lcudart

# Target
TARGET = cublas_test

# Source files
SRCS = cublas_test.cu

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRCS)
	$(NVCC) -o $@ $^ $(CFLAGS)

# Rule to run the target
run: $(TARGET)
	./$(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)
