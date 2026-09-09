# vllm.radiance

Runtime modules and HIP kernel sources for the AMD Radeon AI PRO R9700
(gfx1201 / RDNA4), ported from drwolfen/radiance-vllm-r9700.

Layout:

- `vllm/radiance/*.py` — the runtime suite. These sat in site-packages in the
  upstream fork and imported each other by bare top-level names
  (`import radiance_kernels`); here they form a package with relative
  imports. Compiled extension modules (`radiance_mxfp4_fp8`, ...) are built
  from `kernels-src/` at image build and loaded through a sys.path insert on
  this directory, so a JIT-built `.so` resolves beside its Python module.
- `vllm/radiance/kernels-src/` — HIP / header sources compiled at image build
  (`radiance_mxfp4_fp8.hip`, `radiance_autoround.hip`,
  `radiance_escha.hip`, `radiance_paroquant.hip`) plus the standalone
  `radiance_amdsmi.py` interpreter-startup module consumed via a `.pth` hook
  in the image.

The dispatcher entry point is `radiance_kernels.install_all()`, invoked by
the vLLM plugin loader (see `vllm/plugins/__init__.py`) when
`RADIANCE_RUNTIME_HOOKS=1`. Individual hooks are env-gated inside the
modules (`RADIANCE_USE_R4D`, `RADIANCE_PRESHUFFLE`, `RADIANCE_MXFP4_W4A8`,
`RADIANCE_DYNAMIC_DRAFT`, ...).