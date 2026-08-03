# PGXS build for the RGI-backed GPU OLTP FDW.
# Links librgioltp.so (../librgioltp.so) and the CUDA runtime.
MODULE_big = pg_rgi_fdw
OBJS       = pg_rgi_fdw.o pg_gpu_service.o
EXTENSION  = pg_rgi_fdw
DATA       = pg_rgi_fdw--1.0.sql

ENGINE_DIR := $(abspath $(CURDIR)/..)
CUDA_LIB   ?= /usr/local/cuda/lib64

PG_CPPFLAGS = -I$(ENGINE_DIR)/engine
SHLIB_LINK  = -L$(ENGINE_DIR) -lrgioltp -L$(CUDA_LIB) -lcudart \
              -Wl,-rpath,$(ENGINE_DIR) -Wl,-rpath,$(CUDA_LIB)

PG_CONFIG  ?= pg_config
PGXS       := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
