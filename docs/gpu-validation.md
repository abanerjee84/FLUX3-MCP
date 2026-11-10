# GPU validation

RTX 3060 12 GB; official distilled INT8 ConvRot transformer with NF4 text encoder.
All model tensors audited on CUDA. No prompt upsampler, ComfyUI application, or CPU/disk offloading.

| Case | Seconds | PyTorch reserved MiB | Driver dedicated MiB | Driver shared MiB |
|---|---:|---:|---:|---:|
| generate-512 | 171.28 | 9160.0 | 9291.23 | 78.0 |
| edit-512 | 10.78 | 9316.0 | 9451.23 | 78.0 |
| generate-768 | 12.48 | 9514.0 | 9649.23 | 78.0 |
| generate-1024 | 23.38 | 10074.0 | 10209.23 | 78.0 |
| edit-1024 | 54.97 | 10662.0 | 10797.24 | 78.0 |

The first generation includes cold model loading and encoder quantization. Other cases reuse the models.
A separate 256 MiB CUDA allocation test measured 76 MiB of Windows shared GPU memory before model inference.
The model cases measured about 78 MiB shared, rather than gigabytes of model spill. The shared counter includes driver mappings.

Visual inspection: a red teapot was generated and recolored blue while preserving the scene.
Live MCP validation passed for stdio image editing and HTTP image preview delivery; see benchmarks.json.
Eight automated service/MCP tests passed. Ruff and Python compilation passed.

Sanitized measurements and live MCP summary: [benchmarks.json](benchmarks.json). Runtime files, IDs and images are not published.
