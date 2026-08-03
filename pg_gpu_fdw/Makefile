# PGXS build for the GPU OLTP foreign data wrapper.
# Links the engine shared lib (../libgpuoltp.so) and the CUDA runtime.
MODULE_big = pg_gpu_fdw
OBJS       = pg_gpu_fdw.o
EXTENSION  = pg_gpu_fdw
DATA       = pg_gpu_fdw--1.0.sql

# Absolute path to gpu_oltp/ (holds libgpuoltp.so and engine/ headers).
ENGINE_DIR := $(abspath $(CURDIR)/..)
CUDA_LIB   ?= /usr/local/cuda/lib64

PG_CPPFLAGS = -I$(ENGINE_DIR)/engine
# rpath so the backend can locate libgpuoltp.so + libcudart at run time.
SHLIB_LINK  = -L$(ENGINE_DIR) -lgpuoltp -L$(CUDA_LIB) -lcudart \
              -Wl,-rpath,$(ENGINE_DIR) -Wl,-rpath,$(CUDA_LIB)

PG_CONFIG  ?= pg_config
PGXS       := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
