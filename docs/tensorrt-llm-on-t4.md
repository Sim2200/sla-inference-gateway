# TensorRT-LLM on the free T4: what was tried and why it stopped

The plan was a TensorRT-LLM tier next to the existing ones, serving a small instruction model
(Qwen2.5-0.5B-Instruct) in FP16 and INT8 weight-only on a Kaggle T4, with tokens/s and p95 under
the same load generator. It did not get to a measurement. This page records why, so the claim is
not made and the work is not repeated.

Environment: Kaggle GPU image of 2026-10-13 (Ubuntu 24.04, Python 3.13, CUDA 12.8 toolkit, driver
580.178.04, Tesla T4 with compute capability 7.5), TensorRT-LLM 1.2.1 from pypi.nvidia.com,
installed into a micromamba Python 3.12 environment. The probe script is `kaggle/trtllm_probe.py`.

## What it took to get the library running at all

1. `pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com` (8.5 minutes) installs a
   CUDA 13 build. The image's toolkit is CUDA 12.8, so `import tensorrt_llm` failed with
   `libcublasLt.so.13: cannot open shared object file`.
2. The CUDA 13 runtime libraries come from pip under names without a `-cu13` suffix
   (`nvidia-cublas>=13`, `nvidia-cufft`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-cusparse`,
   `nvidia-nvjitlink`; only `nvidia-cudnn-cu13` and `nvidia-nccl-cu13` keep the suffix; the
   `-cu13` names for the others are deprecated stubs whose install fails). With every
   `site-packages/nvidia/*/lib` on `LD_LIBRARY_PATH`, the native bindings load.
3. TensorRT-LLM pins `numpy<2` while the `datasets` dependency pulls a `pyarrow` that needs
   NumPy 2; `pip install "pyarrow<21"` resolves it.
4. The library spawns MPI workers that re-import the calling script. Configuration has to come
   from environment variables and all work must sit behind `if __name__ == "__main__"`, or the
   worker dies and the parent hangs until killed.
5. INT8 weight-only (`QuantConfig(quant_algo=QuantAlgo.W8A16)`) is accepted only by the TensorRT
   engine backend (`tensorrt_llm._tensorrt_engine.LLM`), not by the default PyTorch backend.

After these, `TensorRT LLM inited` is printed and the GPU is recognised.

## Where it stops

- Default build: the engine build aborts in the fused multi-head attention runner:
  `Assertion failed: Unsupported architecture (../tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.cpp:87)`.
  Same for FP16 and INT8.
- Fused context attention disabled (`plugin_config.context_fmha = False`) with packed input
  kept on: the engine builds (FP16 in 55 s, INT8 weight-only in 45 s, 620 MiB) and loads, and the
  first generate call fails in the decode attention kernel:
  `CUDA runtime error in cudaOccupancyMaxActiveBlocksPerMultiprocessor(... mmha::masked_multihead_attention_kernel ...)`,
  which is what a kernel that was not compiled for the device's architecture reports; the process
  then hangs until the timeout.
- With packed input also disabled the engine builds but the runtime refuses it:
  `TrtGptModelInflightBatching requires GPT attention/Mamba Conv 1d plugin with packed input and paged KV cache`.

So the 1.2.1 wheels ship attention kernels for Ampere and newer only. Older documentation listed
Turing with FP16 and INT8 as community-supported; that is no longer the case in the shipped wheels,
and there is no free GPU on Kaggle or Colab newer than Turing for this account.

## What this means for the project

- No TensorRT-LLM tier, no FP16 vs INT8 token throughput numbers, and nothing about
  TensorRT-LLM on the resume. The Triton Inference Server tier (results section 8) is the LLM-serving
  neighbour this project does have.
- The path that would work: an Ampere or newer GPU (L4, A10, A100, RTX 30/40 series). On such a
  machine the probe script as written should run to the measurement, since everything before the
  attention kernels was resolved.
