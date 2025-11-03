ROCM_PATH = /opt/rocm-6.4.0
CXX = $(ROCM_PATH)/bin/hipcc

# Cray libfabric paths
LIBFABRIC_INCLUDE = /opt/cray/libfabric/2.1/include
LIBFABRIC_LIB = /opt/cray/libfabric/2.1/lib64

# Cray PMI paths
PMI_INCLUDE = /opt/cray/pe/pmi/default/include
PMI_LIB = /opt/cray/pe/pmi/default/lib

# ROCm/HIP paths
HIP_INCLUDE = $(ROCM_PATH)/include
HIP_LIB = $(ROCM_PATH)/lib

# Compiler flags for hipcc
# Note: hipcc needs -x hip to recognize GPU kernels
CXXFLAGS = -std=c++11 -Wall -Wextra
INCLUDES = -I$(LIBFABRIC_INCLUDE) -I$(PMI_INCLUDE)
LDFLAGS = -L$(LIBFABRIC_LIB) -Wl,-rpath,$(LIBFABRIC_LIB) \
          -L$(PMI_LIB) -Wl,-rpath,$(PMI_LIB)
LIBS = -lfabric -lpmi2

# GPU architecture
GPU_ARCH = gfx90a

# Targets
TARGETS = test test_2 rank rank-amdgpu rank-amdgpu-gda rank-nvidia-gda benchmark-mpi benchmark-mpi-rma

all: $(TARGETS)

# Single-node test
test: test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $^ $(LIBS)

# Two-node GPU-initiated RDMA test (pure libfabric, no MPI)
test_2: test_2.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) --offload-arch=$(GPU_ARCH) \
		-o $@ $^ $(LIBS)

rank: rank.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)

rank-amdgpu: rank-amdgpu.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $^ $(LIBS)

rank-amdgpu-gda: rank-amdgpu-gda.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $^ $(LIBS)

amdgpu-test: amdgpu-test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $^ $(LIBS)

# MPI GPU-Direct benchmark
benchmark-mpi: benchmark-mpi.cpp
	CC -std=c++11 -D__HIP_PLATFORM_AMD__ -I$(HIP_INCLUDE) -L$(HIP_LIB) -Wl,-rpath,$(HIP_LIB) -o $@ $^ -lamdhip64 -lmpi -lmpi_gtl_hsa

# MPI RMA (MPI_Put) GPU-Direct benchmark
benchmark-mpi-rma: benchmark-mpi-rma.cpp
	CC -std=c++11 -D__HIP_PLATFORM_AMD__ -I$(HIP_INCLUDE) -L$(HIP_LIB) -Wl,-rpath,$(HIP_LIB) -o $@ $^ -lamdhip64 -lmpi -lmpi_gtl_hsa

clean:
	rm -f $(TARGETS) *.core

.PHONY: all clean
