# Build the GPU OLTP engine (shared lib for the FDW) and the microbench.
# RTX 4060 is Ada (sm_89). Adjust ARCH for other GPUs.
ARCH    ?= sm_89
NVCC    ?= nvcc
NVFLAGS := -O3 -arch=$(ARCH)

ENGINE  := engine/gpu_oltp_engine.cu

all: libgpuoltp.so oltp_bench

# Shared library the Postgres FDW links against.
libgpuoltp.so: $(ENGINE) engine/gpu_oltp_engine.h
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared -o $@ $(ENGINE)

# Standalone microbench (Day 1 deliverable).
oltp_bench: $(ENGINE) engine/gpu_oltp_engine.h
	$(NVCC) $(NVFLAGS) -DBUILD_BENCH -o $@ $(ENGINE)

# Convenience: run a small sweep for the locking study.
sweep: oltp_bench
	@echo "== uniform, 50% writes, three schemes =="
	@./oltp_bench 16777216 4194304 4194304 0 0.0 50
	@./oltp_bench 16777216 4194304 4194304 1 0.0 50
	@./oltp_bench 16777216 4194304 4194304 2 0.0 50
	@echo "== zipfian theta=0.99 (contention), three schemes =="
	@./oltp_bench 16777216 4194304 4194304 0 0.99 50
	@./oltp_bench 16777216 4194304 4194304 1 0.99 50
	@./oltp_bench 16777216 4194304 4194304 2 0.99 50

clean:
	rm -f libgpuoltp.so oltp_bench

.PHONY: all sweep clean
