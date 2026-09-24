"""Record the environment and run a GPU sanity check. Writes results/env.json."""
import json
import os
import platform
import subprocess
import time

import datasets
import torch
import transformers

os.makedirs("results", exist_ok=True)
assert torch.cuda.is_available(), "CUDA not available"

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
a @ a
torch.cuda.synchronize()
t = time.time()
for _ in range(20):
    a @ a
torch.cuda.synchronize()
tflops = 2 * 4096**3 / ((time.time() - t) / 20) / 1e12

c = torch.randn(512, 512, device="cuda")
max_err = (c @ c - (c.cpu() @ c.cpu()).cuda()).abs().max().item()

smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
).stdout.strip()

env = {
    "os": platform.platform(),
    "python": platform.python_version(),
    "cpu": platform.processor() or platform.machine(),
    "cpu_threads": os.cpu_count(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_capability": torch.cuda.get_device_capability(0),
    "nvidia_smi": smi,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "fp16_matmul_tflops": round(tflops, 1),
    "fp32_gpu_vs_cpu_max_abs_err": max_err,
}
print(json.dumps(env, indent=2))
json.dump(env, open("results/env.json", "w"), indent=2)
