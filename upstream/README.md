# upstream/ — a patch to RobustGPUIndexing (not a fork)

This directory holds **one small patch** against RobustGPUIndexing (RGI),
the GPU index library by **Hyoungjoo Kim (Carnegie Mellon University),
Apache-2.0**. RGI is a *dependency* of this project, not part of it — clone
it separately (`../RobustGPUIndexing/`); do not vendor its source here.

## The patch

`rgi_launch_geometry_cache.patch` — caches per-instantiation launch geometry
in `kernels.hpp`'s `launch_batch_kernel` instead of calling
`cudaGetDeviceProperties` on **every** launch. On the GH200 driver stack
that call cost **~875 µs per launch** and silently dominated every
launch-mode measurement until patched. The patch replaces it with a single
`cudaDeviceGetAttribute`.

Apply against an RGI checkout with:

```bash
cd ../RobustGPUIndexing && git apply /path/to/rgi_launch_geometry_cache.patch
```

## Status / courtesy

This is a **measurement fix to report upstream to Hyoungjoo**, not a
divergent fork. RGI is under submission (VLDB); **coordinate with him before
publishing anything RGI-derived**, including this patch, if this repo goes
public. The full patched file is intentionally **not** kept here — only the
diff — so RGI's source is not republished.
