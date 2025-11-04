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
TARGETS = test test_2 rank rank-amdgpu rank-amdgpu-gda rank-nvidia-gda benchmark-mpi benchmark-mpi-rma amdgpu-test ofi-comp gda-comp mpi-comp

all: $(TARGETS)

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Build all targets"
	@echo "  amdgpu-test   - GPU-Driven RDMA with DWQ (includes SAXPY computation)"
	@echo "  ofi-comp      - CPU-driven RDMA baseline (includes SAXPY computation)"
	@echo "  gda-comp      - GPU-Driven RDMA comparison test (includes SAXPY computation)"
	@echo "  mpi-comp      - MPI GPU-Direct communication (includes SAXPY computation)"
	@echo "  comp-tests    - Build gda-comp, ofi-comp, and mpi-comp for comparison"
	@echo "  clean         - Remove all built targets"
	@echo ""
	@echo "Usage examples:"
	@echo "  make comp-tests                    # Build all comparison tests"
	@echo "  make gda-comp                      # Build GPU-driven test only"
	@echo "  make ofi-comp                      # Build CPU-driven test only"
	@echo "  make mpi-comp                      # Build MPI test only"

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

# GPU-Driven RDMA test with Deferred Work Queue (DWQ)
# Includes GPU computation (SAXPY) before and after communication
amdgpu-test: amdgpu-test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $^ $(LIBS)

gda-comp: gda-comp.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $^ $(LIBS)


# CPU-driven RDMA test (comparison baseline)
# Includes same GPU computation (SAXPY) before and after communication
ofi-comp: ofi-comp.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) --offload-arch=$(GPU_ARCH) -o $@ $^ $(LIBS)

# MPI GPU-Direct benchmark
benchmark-mpi: benchmark-mpi.cpp
	CC -std=c++11 -D__HIP_PLATFORM_AMD__ -I$(HIP_INCLUDE) -L$(HIP_LIB) -Wl,-rpath,$(HIP_LIB) -o $@ $^ -lamdhip64 -lmpi -lmpi_gtl_hsa

# MPI RMA (MPI_Put) GPU-Direct benchmark
benchmark-mpi-rma: benchmark-mpi-rma.cpp
	CC -std=c++11 -D__HIP_PLATFORM_AMD__ -I$(HIP_INCLUDE) -L$(HIP_LIB) -Wl,-rpath,$(HIP_LIB) -o $@ $^ -lamdhip64 -lmpi -lmpi_gtl_hsa

# MPI GPU-Direct communication benchmark with computation (SAXPY)
# Includes same GPU computation (SAXPY) before and after communication
# Note: Uses hipcc instead of CC to compile GPU kernels
mpi-comp: mpi-comp.cpp
	$(CXX) $(CXXFLAGS) --offload-arch=$(GPU_ARCH) -I$(CRAY_MPICH_DIR)/include -L$(CRAY_MPICH_DIR)/lib -Wl,-rpath,$(CRAY_MPICH_DIR)/lib -o $@ $^ -lmpi -lmpi_gtl_hsa

comp-tests: gda-comp ofi-comp mpi-comp
	@echo "Built all comparison tests: gda-comp, ofi-comp, and mpi-comp"

clean:
	rm -f $(TARGETS) *.core

.PHONY: all clean help gda-comp cpu-comp comp-tests
