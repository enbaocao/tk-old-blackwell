# GPU Selection: 4090, A100, H100
GPU_TARGET=H100

# Compiler
NVCC?=nvcc

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas -lgomp -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype # H100
TARGET=matmul
SRC=matmul.cu
FP8_TARGET=matmul_fp8
FP8_SRC=matmul_fp8.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

$(FP8_TARGET): $(FP8_SRC)
	$(NVCC) $(FP8_SRC) $(NVCCFLAGS) -o $(FP8_TARGET)

# Clean target
clean:
	rm -f $(TARGET) $(FP8_TARGET)