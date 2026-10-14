"""Feasibility probe: can TensorRT-LLM serve a small LLM on Kaggle's free T4 (Turing, sm_75)?

Pushed as a private Kaggle script kernel (GPU, internet on). The result, after nine iterations of
this file on 2026-10-13/14, is recorded in docs/tensorrt-llm-on-t4.md: engines build and load, but
generation fails because the attention kernels are not built for sm_75. Kept so the finding can be
reproduced; see the doc for the environment details each step depends on.
"""
import json, os, subprocess, sys, time

def sh(cmd, timeout=3000):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout)
    return {"cmd": cmd[:200], "rc": r.returncode, "s": round(time.time() - t0, 1), "out": (r.stdout + r.stderr)[-2500:]}

rep = {}
rep["gpu"] = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; nvcc --version | tail -1")
rep["env"] = sh("curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba && /tmp/bin/micromamba create -y -q -r /tmp/mm -p /tmp/py312 -c conda-forge python=3.12 2>&1 | tail -2")
PY = "/tmp/py312/bin/python"
rep["pip_trtllm"] = sh(f"{PY} -m pip install -q tensorrt_llm --extra-index-url https://pypi.nvidia.com 2>&1 | tail -5", timeout=3500)
# the wheel is a CUDA 13 build; the image has the CUDA 12.8 toolkit, so supply the CUDA 13 runtime libraries from pip
# (CUDA 13 wheels drop the -cu13 suffix except cuDNN and NCCL)
rep["pip_cu13"] = sh(f"{PY} -m pip install -q 'nvidia-cublas>=13,<14' 'nvidia-cufft>=12,<13' 'nvidia-curand>=10.4' 'nvidia-cusolver>=12,<13' 'nvidia-cusparse>=12.6' 'nvidia-nvjitlink>=13,<14' nvidia-cudnn-cu13 nvidia-nccl-cu13 'nvidia-cuda-nvrtc>=13,<14' 2>&1 | tail -4", timeout=1500)
rep["libs"] = sh("find /tmp/py312/lib/python3.12/site-packages/nvidia -name 'lib*.so.1[23]' | sed 's#.*/##' | sort | tr '\\n' ' '")
LIBS = subprocess.run("find /tmp/py312/lib/python3.12/site-packages/nvidia -type d -name lib | tr '\\n' ':'", shell=True, capture_output=True, text=True).stdout
os.environ["LD_LIBRARY_PATH"] = LIBS + os.environ.get("LD_LIBRARY_PATH", "")
rep["ld_path"] = LIBS[:800]
rep["pip_fix"] = sh(f"{PY} -m pip install -q 'pyarrow<21' 2>&1 | tail -2; {PY} -c 'import numpy, pyarrow; print(numpy.__version__, pyarrow.__version__)'")
VERS = f"{PY} -c 'import tensorrt_llm, torch, tensorrt; print(tensorrt_llm.__version__, torch.__version__, torch.version.cuda, tensorrt.__version__, torch.cuda.get_device_capability())'"
rep["versions"] = sh(VERS)
test = r'''
import json, os, sys, time, torch
os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")


def main():
    # TensorRT-LLM spawns MPI workers that re-import this file, so the precision comes from the
    # environment (inherited) and all work sits behind the __main__ guard
    name = os.environ["PROBE_PRECISION"]
    print("import...", flush=True)
    from tensorrt_llm import SamplingParams
    from tensorrt_llm.llmapi import QuantConfig, QuantAlgo
    try:
        from tensorrt_llm._tensorrt_engine import LLM  # the TensorRT engine backend
    except ImportError:
        from tensorrt_llm import LLM
    from tensorrt_llm import BuildConfig
    kw = {"quant_config": QuantConfig(quant_algo=QuantAlgo.W8A16)} if name == "int8_weight_only" else {}
    # the fused multi-head attention kernels refuse sm_75 ("Unsupported architecture", fmhaRunner.cpp);
    # try the build without them
    bc = BuildConfig(max_batch_size=8, max_seq_len=512, max_input_len=256, max_num_tokens=2048)
    # the runtime's inflight batching needs packed input and a paged KV cache, so only the fused
    # attention kernels are turned off (they are what rejects sm_75)
    bc.plugin_config.context_fmha = False
    bc.plugin_config.use_paged_context_fmha = False
    bc.plugin_config.remove_input_padding = True
    bc.plugin_config.paged_kv_cache = True
    print("building", name, "with", LLM.__module__, "context_fmha=False remove_input_padding=True", flush=True)
    t0 = time.time()
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16", build_config=bc, **kw)
    build_s = time.time() - t0
    print("built in", round(build_s, 1), "s", flush=True)
    prompts = ["Explain what a wafer map is in two sentences."] * 4
    sp = SamplingParams(max_tokens=128, temperature=0.0)
    llm.generate(prompts[:1], sp)
    t0 = time.time(); res = llm.generate(prompts, sp); dt = time.time() - t0
    toks = sum(len(r.outputs[0].token_ids) for r in res)
    out = {"build_s": round(build_s, 1), "tokens": toks, "seconds": round(dt, 2), "tokens_per_s": round(toks / dt, 1), "sample": res[0].outputs[0].text[:200]}
    print(json.dumps(out), flush=True)
    open(f"/kaggle/working/trtllm_{name}.json", "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
'''
open("/tmp/test.py", "w").write(test)
for name in ("fp16", "int8_weight_only"):
    t0 = time.time()
    # live output into the kernel log; 20 minutes each
    r = subprocess.run(f"cd /tmp && HF_HOME=/tmp/hf PROBE_PRECISION={name} timeout 1000 {PY} -u /tmp/test.py", shell=True, text=True)
    rep[f"run_{name}"] = {"rc": r.returncode, "s": round(time.time() - t0, 1), "note": "124 = timed out after 1200 s; output is in the kernel log"}
    print(f"== {name}: rc {r.returncode} in {time.time() - t0:.0f} s", flush=True)
print(json.dumps(rep, indent=1))
open("/kaggle/working/trtllm_probe.json", "w").write(json.dumps(rep, indent=1))
