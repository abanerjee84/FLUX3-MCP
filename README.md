# FLUX3-MCP

Local FLUX 3 Image Klein generation and editing through **MCP, a global CLI, or HTTP**. One persistent worker owns the GPU models and serves every connected client through a shared queue.

Built and tested for an **NVIDIA RTX 3060 with 12 GB VRAM on Windows**:

- Official distilled **INT8 ConvRot image transformer**, running in Diffusers.
- **Qwen3-VL-4B text encoder in four-bit NF4**, plus the FLUX.2 VAE.
- The host agent supplies the final prompt. There is **no prompt upsampler**.
- No ComfyUI application, CPU model offloading, or automatic switch to another GPU.
- MCP over stdio and Streamable HTTP, plus REST and a command-line interface.

**Model weights are not included in this repository.** A separate setup command downloads the pinned files from Hugging Face into an ignored local directory.

For a first installation, follow the [step-by-step guide](#installation) or give an agent the [installation brief](#installation-instructions-for-an-agent). For daily use, jump to [common commands](#common-commands-start-to-stop).

## Contents

- [Measured results](#measured-results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Installation instructions for an agent](#installation-instructions-for-an-agent)
- [Common commands: start to stop](#common-commands-start-to-stop)
- [Generate and edit](#generate-and-edit)
- [Complete CLI reference](#complete-cli-reference)
- [Agent harness integration](#agent-harness-integration)
- [MCP tools](#mcp-tools)
- [HTTP API](#http-api)
- [Architecture and ConvRot loading](#architecture-and-convrot-loading)
- [GPU policy and request limits](#gpu-policy-and-request-limits)
- [Configuration and local files](#configuration-and-local-files)
- [Pinned models and dependencies](#pinned-models-and-dependencies)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Updating and removing the installation](#updating-and-removing-the-installation)
- [Attribution and model licenses](#attribution-and-model-licenses)

## Measured results

Single-image runs on RTX 3060 12 GB, driver 610.88, Python 3.12.9, CUDA PyTorch 2.14.1+cu130, 16 steps, seed 7, INT8 ConvRot transformer and NF4 encoder. The scene was a red teapot, edited to blue. Outputs were visually inspected.

| Operation | Resolution | Seconds | PyTorch peak reserved MiB | Driver peak dedicated MiB |
|---|---|---:|---:|---:|
| Generation, including cold loading and encoder quantization | 512 x 512 | 171.28 | 9,160 | 9,291.23 |
| Edit, one reference, models already loaded | 512 x 512 | 10.78 | 9,316 | 9,451.23 |
| Generation, models already loaded | 768 x 768 | 12.48 | 9,514 | 9,649.23 |
| Generation, models already loaded | 1024 x 1024 | 23.38 | 10,074 | 10,209.23 |
| Edit, one 1024-pixel reference, models already loaded | 1024 x 1024 | 54.97 | 10,662 | 10,797.24 |

The largest measured dedicated-memory peak was about **10.54 GiB**. Registered model parameters and buffers were audited on CUDA; CPU/disk mappings and offload hooks were rejected. Windows reported 78 MiB shared GPU memory during these runs. A separate small CUDA-allocation check reported 76 MiB without the model pipeline; shared counters include driver mappings and are not equivalent to model offloading.

These are observed runs for one scene, not latency percentiles or a guarantee for every prompt/reference combination. Allocator caches persist between jobs. See the [validation report](docs/gpu-validation.md) and [sanitized measurements](docs/benchmarks.json).

## Requirements

| Item | Requirement |
|---|---|
| OS | Windows x64; installer and global launcher are Windows-specific |
| GPU | RTX 3060 **12 GB**, with enough free memory for the worker |
| NVIDIA driver | Compatible with the bundled CUDA 13.0 PyTorch wheels; tested with 610.88 |
| Python | 3.12; setup installs a managed Python through uv when needed |
| Tools | [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git |
| Storage | About 15 GB of model weights, plus Python packages, download caches, temporary files and outputs |
| Network | Required for initial installation/downloads; inference then reads local model files |
| Hugging Face | Access to both official repositories and authentication if their access controls require it |

Keep system RAM available for loading checkpoints, tokenization, image preprocessing, orchestration and normal CUDA staging. GPU-only model inference does not mean that the loader and operating system use no RAM.

The implementation deliberately requires a GPU name containing `RTX 3060`. Other GPU families are not enabled. A 3060 Ti or 6 GB variant is not the validated 12 GB target. Linux, WSL and macOS are not validated installation targets.

## Installation

Use **PowerShell** for the commands below. Paste one block at a time. You do not need to install Python yourself or activate a virtual environment manually.

Already installed? Go straight to [Common commands: start to stop](#common-commands-start-to-stop). To delegate a new installation, copy the [agent installation instructions](#installation-instructions-for-an-agent).

### 1. Check your GPU and install the two setup tools

This build targets **Windows x64 with an RTX 3060 12 GB**. Check that Windows can see it:

```powershell
nvidia-smi
```

If that command is missing or does not show the intended GPU, install or repair your NVIDIA driver before continuing. See [Requirements](#requirements) for the tested driver and hardware scope.

Check whether Git and uv are already available:

```powershell
git --version
uv --version
```

Install only the tool that is missing, or both if this is a fresh machine:

```powershell
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e
```

Then **close and reopen PowerShell** so the new commands are on PATH. If you do not have WinGet, use the official [Git for Windows](https://git-scm.com/install/windows) and [uv installation](https://docs.astral.sh/uv/getting-started/installation/) instructions.

### 2. Choose a permanent folder and clone the project

These examples use `E:\TOOLS\FLUX3-MCP`. If you do not have an E: drive, replace that location with a permanent folder on your machine. Keep the same location throughout the instructions.

```powershell
New-Item -ItemType Directory -Path E:\TOOLS -Force | Out-Null
git clone https://github.com/abanerjee84/FLUX3-MCP.git E:\TOOLS\FLUX3-MCP
Set-Location E:\TOOLS\FLUX3-MCP
```

If the project is already in that folder, use the existing checkout; do not clone over it. The launcher and agent registrations use absolute paths, so keep the folder in place after installation.

### 3. Install the application

From the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

Let this finish before continuing. It creates `.venv`, installs Python 3.12 when needed, installs CUDA PyTorch and the application dependencies, and adds `flux-image` to your user PATH. The execution-policy override applies only to this PowerShell process.

**This step installs the application, not the model weights.** The next two steps handle account access and the model download.

### 4. Sign in to Hugging Face

Open both model pages in your browser. Sign in and complete an access form if one is shown:

- [Main Klein repository](https://huggingface.co/black-forest-labs/flux-3-image-klein)
- [Official INT8 ConvRot weights repository](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)

These are separate repositories; access to one does not automatically establish access to the other. The second name contains `comfy`, but this application uses only its weight file and does not install ComfyUI.

Sign the local CLI in to the **same Hugging Face account**:

```powershell
.\.venv\Scripts\hf.exe auth login
```

Follow the prompts in your terminal. Keep the token in the local authentication flow; it does not belong in a prompt file or repository file.

### 5. Download the models

```powershell
.\.venv\Scripts\python.exe -m flux_image setup-models
```

Allow time and disk space for about **15 GB of weights**, plus the environment and download caches. Progress is printed while downloading, and large files are checked against their SHA-256 hashes before installation.

If the download is interrupted, run the same command again to resume. If it returns 403, revisit **both** model pages and confirm that the CLI is signed in to the account with access.

The download includes the INT8 image transformer, VAE and source weights needed to quantize the text encoder to NF4. It does not include a prompt upsampler or the base image transformer.

### 6. Connect your agent, if you want MCP tools

Run only the row for the harness you use. Run more than one row if you use several harnesses. These commands work before refreshing PATH:

| Your agent harness | Run from the project folder |
|---|---|
| Codex | `.\.venv\Scripts\python.exe -m flux_image register codex` |
| Claude Code | `.\.venv\Scripts\python.exe -m flux_image register claude-code` |
| Gemini CLI | `.\.venv\Scripts\python.exe -m flux_image register gemini` |
| Another MCP client | `.\.venv\Scripts\python.exe -m flux_image mcp-config` |

For another client, copy the generated configuration into that client's MCP settings. See [Agent harness integration](#agent-harness-integration) for details.

Registration is user-level, so it applies across projects in that harness. Existing unrelated settings are preserved. **Restart the harness** after registration to discover the `local-flux` tools.

### 7. Verify the installation

Open a **new PowerShell window**, then run:

```powershell
flux-image doctor
```

Check the JSON output:

- `inference.ready` should be `true`.
- `missing_files` and `missing_packages` should be empty lists.
- The selected GPU should be your RTX 3060.

`loaded: false` is normal before the first generation. `service_running: false` is also normal after installation: the model server has not been started yet.

If `flux-image` is not recognized, use this from the project folder:

```powershell
.\bin\flux-image.cmd doctor
```

The installation is now ready. Follow [Common commands: start to stop](#common-commands-start-to-stop) for a complete first session.

For an already-authenticated Hugging Face account and a Codex installation, the setup script also offers a combined shortcut:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -DownloadModels -RegisterCodex
```

## Installation instructions for an agent

Copy the following brief into your local coding agent. It describes the installation task, account handoff and verification; it is not a PowerShell script.

```text
Install https://github.com/abanerjee84/FLUX3-MCP on this Windows computer.

Use E:\TOOLS\FLUX3-MCP unless I specify another location. If that drive is
unavailable, choose a permanent user-writable location and tell me its path.

1. Inspect the existing installation first. If the target folder already
   contains this project, check its remote and working tree, preserve local
   changes, and reuse its model files. Avoid duplicate model downloads.

2. Verify Windows x64, an RTX 3060 with 12 GB VRAM, an NVIDIA driver compatible
   with the pinned CUDA wheels, Git and uv. Install missing setup tools using
   their official instructions. Report a hardware mismatch rather than
   changing the project's GPU checks or enabling CPU offloading.

3. Clone the repository into the selected permanent location if needed.
   From that folder, run:
   powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
   The execution-policy override is process-scoped. Check native command
   exit codes.
   The basic setup installs the environment and global CLI, not the models.

4. Use existing Hugging Face authentication if it has the required access.
   If login or acceptance is needed, let me complete the local hf auth login
   flow and access forms for both black-forest-labs/flux-3-image-klein and
   black-forest-labs/flux-3-image-klein-comfy. Keep tokens out of chat, source
   files and logs.

5. From the project folder, run:
   .\.venv\Scripts\python.exe -m flux_image setup-models
   Let downloads and hash verification complete. Resume an interrupted download rather than
   deleting existing weights. Retain the pinned model revisions, official
   INT8 ConvRot transformer and NF4 encoder. Do not add a prompt upsampler.

6. Register the harness I am using at user scope. From the project folder, run:
   .\.venv\Scripts\python.exe -m flux_image register HARNESS
   Replace HARNESS with codex, claude-code, or gemini. For another client, run:
   .\.venv\Scripts\python.exe -m flux_image mcp-config
   Provide its client-specific setup. Preserve unrelated configuration and
   inspect a conflicting local-flux entry before changing
   it. Explain that the harness must restart to load the registration.

7. Verify flux-image doctor using the new executable path if the current
   terminal has stale PATH. Confirm the data directory, selected GPU,
   inference.ready=true, and empty missing-files/missing-packages lists.
   Check command failures and report unresolved dependency or access errors.
   Do not describe installation as successful while these checks fail.

8. Finish with the image server stopped unless I explicitly request it
   running. Leave image generation and GPU validation for a requested smoke
   test. Report the installation path, registered harness, verification
   results, and the commands to start and stop the server.

Keep models, .venv, generated images and local credentials out of Git.
```

## Common commands: start to stop

After installation, open a new PowerShell window. These commands work from any folder. Output paths below are relative to your current folder, so choose a folder where you want to save images.

### 1. Start the server

```powershell
flux-image start
```

This starts the background API/MCP service. It does not load the models yet. The first image request loads them; later requests reuse them.

### 2. Check readiness

```powershell
flux-image doctor
```

Look for `inference.ready: true` and `service_running: true`. `loaded` becomes true after a successful model load.

### 3. Generate and save an image

```powershell
flux-image generate --prompt "Studio photograph of one matte red ceramic teapot on a gray tabletop, curved handle on the left, short spout on the right, fitted lid, white background and soft light from the upper left." --width 768 --height 768 --seed 7 --out teapot-red.png
```

`--out` waits for the image and saves it. For a longer caption, use `--prompt-file prompt.txt` instead of `--prompt`.

### 4. Edit the image and save the result

```powershell
flux-image edit --image teapot-red.png --prompt "The same studio photograph with one matte cobalt blue ceramic teapot on the gray tabletop, curved handle on the left, short spout on the right, fitted lid, the same white background and soft light from the upper left." --width 768 --height 768 --seed 7 --out teapot-blue.png
```

You can now inspect `teapot-red.png` and `teapot-blue.png`. For higher resolution, set both dimensions to 1024; it takes more time and GPU memory.

### 5. Submit a background job and check it

Use this mode when you want the job ID immediately instead of waiting inside `generate`:

```powershell
$imageJob = flux-image generate --prompt "Studio photograph of one red ceramic teapot on a pale gray table, white background, soft light from the left and a gentle contact shadow." --width 768 --height 768 --seed 21 | ConvertFrom-Json
$imageJobId = $imageJob.job_id

flux-image status $imageJobId
flux-image wait $imageJobId --timeout 600
flux-image download $imageJobId --out teapot-background-job.png
```

If you decide to cancel an unfinished job, use `flux-image cancel $imageJobId` instead of the wait/download steps. A client wait timeout does not cancel the job.

### 6. Get help when needed

```powershell
flux-image --help
flux-image generate --help
flux-image edit --help
```

### 7. Stop the server and end the session

Closing the terminal alone leaves the background service running. This command stops it and releases its model process and GPU memory:

```powershell
flux-image stop
```

A later CLI generation/edit request or an MCP tool call that needs the backend can start the service again automatically.

## Generate and edit

### Generate

Write a UTF-8 prompt file containing a complete scene caption:

```powershell
@'
Studio product photograph of one matte red ceramic teapot on a pale gray
tabletop, centered in a three-quarter view. A round body, curved handle on the
left, short spout on the right, and fitted lid. White seamless background.
Soft diffused light from the upper left, a gentle contact shadow, realistic
ceramic texture and crisp edges. No lettering or other objects.
'@ | Set-Content -Encoding utf8 prompt.txt

flux-image generate --prompt-file prompt.txt --width 1024 --height 1024 --seed 7 --out result.png
```

`--out` waits and saves a copy. The original PNG and JSON metadata sidecar stay in `data/outputs/`. The PNG also embeds the prompt, seed, model and memory metadata.

Inline prompts work too:

```powershell
flux-image generate --prompt "Studio photograph of a red ceramic teapot on a gray table, soft light from the left, plain white background." --out result.png
```

### Edit

```powershell
flux-image edit --image result.png --prompt-file edit.txt --width 1024 --height 1024 --seed 7 --out edited.png
```

Describe the complete desired result in `edit.txt`, incorporating the change. For this example, describe the same scene with a cobalt blue teapot. The host agent should inspect the reference before writing that caption. There is no local LLM that expands a short edit instruction.

Repeat `--image` for references:

```powershell
flux-image edit --image subject.png --image background.png --prompt-file composition.txt --out combined.png
```

The CLI accepts local file paths. MCP also accepts uploaded image IDs and previous successful job IDs.

### Submit without waiting

```powershell
flux-image generate --prompt-file prompt.txt --request-id poster-v1
```

Omit `--out` and `--wait` to receive a JSON job record immediately. Use its `job_id`:

```powershell
flux-image status JOB_ID
flux-image wait JOB_ID --timeout 600
flux-image download JOB_ID --out result.png
flux-image cancel JOB_ID
```

A stable request ID makes retries of the identical request return the original job and seed. A different request using that ID is rejected. Closing a client or timing out a wait does not cancel inference.

## Complete CLI reference

| Command | Behavior |
|---|---|
| `flux-image --help` | List commands |
| `flux-image doctor` | Check GPU inventory, models, dependencies and live service status |
| `flux-image start` | Start the shared daemon if needed |
| `flux-image serve` | Run the singleton service in the foreground |
| `flux-image stop` | Shut down this service and release its GPU process |
| `flux-image generate ...` | Submit text-to-image generation |
| `flux-image edit --image FILE ...` | Submit editing with one to four references |
| `flux-image status JOB_ID` | Read progress, status, errors and output metadata |
| `flux-image wait JOB_ID [--timeout SECONDS]` | Wait for completion; default 1800 seconds |
| `flux-image cancel JOB_ID` | Cancel pending work or request cancellation at an inference boundary |
| `flux-image download JOB_ID --out FILE` | Save the completed full-resolution PNG |
| `flux-image setup-models` | Download/resume and verify pinned weights |
| `flux-image install-cli` | Install the generated user-PATH launcher |
| `flux-image register codex` | Register MCP at Codex user scope |
| `flux-image register claude-code` | Register MCP at Claude Code user scope |
| `flux-image register gemini` | Register MCP at Gemini CLI user scope |
| `flux-image register-codex` | Alias for Codex registration |
| `flux-image mcp-config` | Print generic stdio MCP configuration |
| `flux-image mcp-config --http` | Print HTTP MCP endpoint information |
| `flux-image mcp` | Run the stdio adapter, normally launched by a harness |

Shared generation/editing options:

| Option | Meaning | Default |
|---|---|---|
| `--prompt TEXT` | Inline caption, mutually exclusive with `--prompt-file` | One prompt input required |
| `--prompt-file FILE` | Read a UTF-8 caption | One prompt input required |
| `--width INTEGER` | Output width | 768 |
| `--height INTEGER` | Output height | 768 |
| `--steps INTEGER` | Sampling steps | 16 |
| `--seed INTEGER` | Seed from 0 to 4294967295 | Random |
| `--request-id TEXT` | Retry identity for an identical request | Unset |
| `--wait` | Wait for completion | Off |
| `--timeout SECONDS` | Wait limit, not an inference cancellation timer | 1800 |
| `--out FILE` | Save a copy after success; implies waiting | Unset |

`edit` additionally requires `--image FILE`; repeat it up to four times. Use `flux-image COMMAND --help` for specific help.

Results are JSON on stdout. Reported command errors are JSON on stderr and return nonzero exit codes. A submission that has not waited cannot report a later inference failure: inspect its final status.

## Agent harness integration

### Automatic registration

```powershell
flux-image register codex
flux-image register claude-code
flux-image register gemini
```

| Harness | User-level file |
|---|---|
| Codex | `~/.codex/config.toml`, or `$CODEX_HOME/config.toml` |
| Claude Code | `~/.claude.json`, top-level `mcpServers` |
| Gemini CLI | `~/.gemini/settings.json`, top-level `mcpServers` |

Registration adds only `local-flux`, backs up existing configuration before writing, preserves unrelated settings, and refuses to replace a conflicting entry under that name. Restart the harness to discover its tools. Registration does not bypass tool approvals or sandbox rules.

### Other stdio clients

Print paths for the actual installation:

```powershell
flux-image mcp-config
```

Or adapt [mcp-config.example.json](mcp-config.example.json), replacing both paths:

```json
{
  "mcpServers": {
    "local-flux": {
      "command": "C:/AI/FLUX3-MCP/.venv/Scripts/python.exe",
      "args": ["-m", "flux_image", "mcp"],
      "env": {"FLUX_IMAGE_HOME": "C:/AI/FLUX3-MCP/data"}
    }
  }
}
```

Every adapter forwards calls to the same daemon; it does not load another model. Absolute executable/data paths make registration independent of the active project directory.

### HTTP MCP

Start the service and configure a Streamable HTTP client for `http://127.0.0.1:8765/mcp`. Send `Authorization: Bearer <local-token>` using the token in `data/config.json`. The endpoint uses standard SSE streaming responses, not the deprecated standalone SSE transport. `mcp-config --http` prints the endpoint, not the token.

Clients differ in how they configure headers. Use that client's format for the URL and bearer header. This project creates no public tunnel or remote listener.

### Without MCP

Give the harness shell access to the CLI or integrate REST. A useful persistent instruction is:

> Use `flux-image` for local image generation and editing. Supply a complete target-scene caption. Submit a job, poll it to completion, and inspect the resulting image. The service has no prompt upsampler. Use absolute input/output paths when the working directory is uncertain.

There is no registry that automatically enables a tool in every agent product. Each harness needs an MCP registration, shell access with usage instructions, or HTTP integration. Cloud agents need an explicitly configured connection to this computer; they cannot access its localhost or Windows paths directly.

## MCP tools

| Tool | Inputs | Result |
|---|---|---|
| `generate_image` | `prompt`; optional `width`, `height`, `steps`, `seed`, `client_request_id` | Job record immediately |
| `upload_image` | `image_path`, local to the service computer | Uploaded `image_id` and dimensions |
| `edit_image` | `prompt`, `reference_images`; optional generation settings | Job record immediately |
| `get_image_job` | `job_id`; optional `include_preview=false` | Job record; optional PNG content after success |
| `cancel_image_job` | `job_id` | Current job record |
| `image_service_status` | None | Readiness, selected GPU, placement policy, download progress and latest memory metrics |

MCP `reference_images` accepts uploaded IDs, successful job IDs, or absolute local paths, with one to four references. Clients without shared Windows filesystem access can upload bytes using REST and pass the returned IDs.

Call `generate_image` or `edit_image`, then poll `get_image_job` until `succeeded`, `failed`, or `cancelled`. Request `include_preview=true` after success to attach an image. Previews are limited to 768 pixels on the longest side; saved PNGs and REST downloads retain full resolution.

## HTTP API

All endpoints require the generated local bearer token. The service listens on `127.0.0.1`, default port 8765.

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Readiness and selected GPU |
| POST | `/v1/images` | Upload raw image bytes and receive an image ID |
| POST | `/v1/jobs` | Submit JSON; returns HTTP 202 and a job record |
| GET | `/v1/jobs/{job_id}` | Read progress and results |
| POST | `/v1/jobs/{job_id}/cancel` | Request cancellation |
| GET | `/v1/jobs/{job_id}/image` | Download a successful job's PNG |
| POST | `/admin/shutdown` | Stop a service started by this CLI |
| GET | `/openapi.json` | OpenAPI schema |
| GET | `/docs` | Interactive API documentation, also authenticated |

Start the service first with `flux-image start`. Unlike the CLI and stdio adapter, a raw REST request does not auto-start it. This Python example runs from the checkout:

```python
import json
import time
from pathlib import Path

import httpx

config = json.loads(Path("data/config.json").read_text(encoding="utf-8"))
with httpx.Client(
    base_url=f"http://127.0.0.1:{config['port']}",
    headers={"Authorization": "Bearer " + config["token"]},
    trust_env=False,
    timeout=30,
) as client:
    response = client.post("/v1/jobs", json={
        "prompt": "Studio photograph of a red ceramic teapot, gray tabletop, white background, soft light from the left.",
        "width": 768, "height": 768, "steps": 16, "seed": 7,
        "client_request_id": "example-teapot-v1",
    })
    response.raise_for_status()
    job_id = response.json()["job_id"]

    while True:
        response = client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(1)

    if job["status"] != "succeeded":
        raise RuntimeError(job.get("error") or job["status"])
    response = client.get(f"/v1/jobs/{job_id}/image")
    response.raise_for_status()
    Path("result.png").write_bytes(response.content)
```

For editing, upload each reference as a raw body to `/v1/images`, then put its `image_id` in the job's `reference_images` array. REST job submissions take uploaded IDs, not arbitrary file paths.

Common responses: 400 invalid image, 401 missing/wrong authentication, 403 rejected origin, 404 unknown ID, 409 conflicting request ID or image not ready, 413 oversized upload, 422 invalid settings, and 429 full queue. Loading/CUDA failures happen asynchronously and appear in the job's `error` field.

## Architecture and ConvRot loading

```mermaid
flowchart LR
    A[MCP stdio clients] --> B[Lightweight adapters]
    C[Global CLI] --> D[Local API and shared queue]
    E[HTTP MCP or REST clients] --> D
    B --> D
    D --> F[Single Diffusers worker]
    F --> G[RTX 3060: NF4 encoder + INT8 ConvRot transformer + VAE]
    F --> H[Local PNG and metadata]
```

The CLI or stdio adapter starts the daemon on demand. Process locks prevent duplicate workers for one data directory. A SQLite queue persists jobs, admits up to 32 queued/running requests, and executes one image at a time. Queued jobs survive a restart; interrupted running jobs are marked failed rather than silently retried.

The official INT8 checkpoint is packaged for ComfyUI, so ordinary Diffusers loading is insufficient. [`_convrot.py`](src/flux_image/_convrot.py) bridges it by creating a Diffusers transformer with empty weights, reading INT8 tensors/scales/ConvRot metadata, mapping layer names and splitting fused QKV tensors, and installing custom linear layers backed by standalone `comfy-kitchen` kernels. It converts the remaining floating-point tensors and checks shapes and missing/meta tensors before inference.

The transformer keeps INT8 weights and FP32 scales rather than expanding into a full BF16 model at loading. `comfy-kitchen` is a kernel library; the ComfyUI application is not a dependency. All 258 quantized layer mappings were checked against the target architecture, followed by real generation and editing.

The text encoder converts the host's caption to embeddings and remains necessary without a prompt upsampler. NF4 encoding leaves memory for the image model and VAE. The distilled model defaults to 16 steps without classifier-free guidance, scheduler shift 16.0, and GPU VAE tiles with a 256-pixel sample size.

## GPU policy and request limits

- The GPU UUID is selected before importing PyTorch. Other GPUs are hidden from the worker.
- Automatic selection needs exactly one matching RTX 3060. Missing or ambiguous selection never silently falls back to another device.
- The encoder has an explicit CUDA mapping. There is no `device_map="auto"` or CPU/disk model-offloading path.
- Registered model parameters and buffers must be on CUDA after loading and after each successful inference. CPU/disk mappings and offload hooks are rejected.
- The default PyTorch allocation budget is 10.5 GiB, also bounded by device capacity minus 1.5 GiB. This is not a driver-wide reservation.
- Windows PDH counters sample per-process dedicated/shared GPU memory separately from PyTorch allocated/reserved memory.
- Allocation/residency failures fail the job; the worker does not substitute a CPU path or another quantization format.

A CUDA label and allocator cap alone do not prove physical residency under all Windows driver conditions. Audits, headroom and driver measurements provide evidence for the tested runs. Other GPU consumers, longer captions and reference combinations can change the peak.

| Setting | Current limit |
|---|---|
| Batch | One output image per job |
| Dimensions | 256 through 2048 per axis, multiples of 16 |
| Output area | At most 1,048,576 pixels, including 1024 x 1024 |
| Steps | 4 through 16 |
| References | One through four for editing |
| Reference preprocessing | Combined approximately 1 MP budget divided across references |
| Upload | At most 32 MiB and 16 MP per image |
| Prompt | At most 16,000 characters; backend also enforces a 1024-token caption budget |
| Queue | At most 32 queued/running jobs |

Multi-reference and extreme-aspect-ratio inputs are not exhaustively validated. Measured edits used one reference. Limits are safeguards, not a promise of identical memory use or quality for every request. NF4 encoder results are not expected to be pixel-identical to a BF16 encoder.

## Configuration and local files

The default data root is `<checkout>/data`. Override with `FLUX_IMAGE_HOME`, set consistently for the service, CLI and adapters. Different roots are separate installations; they do not share model instances/jobs and must use non-conflicting ports.

| Path | Contents |
|---|---|
| `.venv/` | Private packages and executables |
| `bin/flux-image.cmd` | Generated launcher with this installation's absolute Python path |
| `data/config.json` | Port, selected GPU UUID, allocation budget and local bearer token |
| `data/models/` | Weights, resume/verification records and model manifest |
| `data/inputs/` | Normalized reference copies |
| `data/outputs/` | Generated PNGs and JSON metadata |
| `data/jobs.sqlite3` | Queue and history |
| `data/service.log` | Daemon log |
| `data/gpu-validation.json` | Hardware report, if validation was run |

The config contains `port` (8765), `gpu_uuid`, `memory_gib` (10.5) and `token`. Stop the service/adapters before changing it. Do not copy it unchanged to a new machine: its UUID and token are installation-specific. Run `doctor` to inspect GPUs.

`FLUX_DOWNLOAD_WORKERS` controls download concurrency, 1 through 8, default 6. Downloads use 16 MiB ranges with up to five attempts per range and final SHA-256 verification. A bad hash is not installed as a usable checkpoint.

Runtime state, weights, generated launchers, logs, secrets, caches and images are ignored by Git. Keep config/tokens/harness backups private. The API requires authentication, validates origins, and binds to loopback; it is not a public multi-user hosting service. Stdio adapters inherit the host user's filesystem access.

## Pinned models and dependencies

| Component | Pinned source |
|---|---|
| Pipeline config, encoder source and VAE | `black-forest-labs/flux-3-image-klein` at `337337621b8526e120f2275d250242f4a6bedd75` |
| Official ConvRot transformer | `black-forest-labs/flux-3-image-klein-comfy` at `5c9b714851dc072b04a23ef6d2d36421f5ccb3a0` |
| Transformer file at that revision | `diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors` |
| Diffusers | `d42ebf9c6e0b46103b74283f5324357a67acaa04` |
| Transformers | `966f59c897c15db8a5e14882d086beba8d6d0b9f` |
| PyTorch / torchvision | `2.14.1+cu130` / `0.29.1+cu130` |
| comfy-kitchen / bitsandbytes | `0.2.37` / `0.50.2` |

The pinned checkpoint has an older name than the current repository files. Keep its revision and filename together rather than replacing only the filename.

[`requirements.lock.txt`](requirements.lock.txt) records the tested environment. Setup installs CUDA wheels from the PyTorch CUDA 13.0 index before the lock file. [`pyproject.toml`](pyproject.toml) defines the package and CLI entry point; [`models.py`](src/flux_image/models.py) pins model revisions. Inference uses local-only loading and sets Hugging Face/Transformers offline flags in the worker.

## Testing

From the checkout:

```powershell
.\scripts\test.ps1
.\.venv\Scripts\ruff.exe check src tests scripts/validate_gpu.py scripts/validate_mcp_live.py
.\.venv\Scripts\python.exe -m compileall -q src
```

Eight automated checks cover serialization, idempotency, queued/running cancellation, restart recovery, authentication, images, input limits, preserving harness configuration, and real MCP sessions over stdio/HTTP. Test backends are injected only by tests; missing models never cause the application to return placeholder images.

Hardware checks perform real inference:

```powershell
.\.venv\Scripts\python.exe scripts/validate_gpu.py --wait-for-models
.\.venv\Scripts\python.exe scripts/validate_mcp_live.py
```

The first waits for models, restarts this service, generates at 512/768/1024 pixels and edits at 512 pixels, writing reports/images under `data/`. The published 1024-pixel edit was an additional run. The second requires that GPU report, performs an edit through stdio MCP, and retrieves an image preview through HTTP MCP. Both consume the GPU. Inspect output quality when validating a new environment.

## Troubleshooting

| Symptom | Action |
|---|---|
| Command not recognized | Open a new terminal; use `bin/flux-image.cmd` or `.venv/Scripts/python.exe -m flux_image` from the checkout. Re-run `install-cli` if needed. |
| Hugging Face 403 | Confirm access to both repositories with the account behind the local token, including the separate `-comfy` repository. Reauthenticate if the account is wrong. |
| Interrupted download | Re-run `setup-models`; completed ranges are reused and hashes are checked. |
| Certificate failure | Setup uses system certificates and downloads use the OS trust store. Install the required corporate/root certificate; do not disable TLS validation. |
| `Inference not ready` | Check `doctor` for missing files/dependencies, then run model setup. |
| GPU UUID unavailable | Inspect `doctor`; stop the service before fixing a stale local UUID. This build targets RTX 3060. |
| CUDA out of memory | Stop other selected-GPU work, reduce dimensions/references and inspect job metrics. No CPU fallback is enabled. |
| Slow first job | Loading and NF4 quantization are included. Keep the daemon running for warm calls. |
| MCP tools absent | Restart the harness after registration; check that `local-flux` is enabled and local execution is permitted. |
| Conflicting registration | Inspect the existing `local-flux` entry and backup; the command refuses to overwrite a different entry. |
| HTTP 401 | Use this installation's generated bearer token, not the Hugging Face token. |
| API docs need authentication | `/docs` is authenticated too. Use an authenticated client or inspect `/openapi.json`. |
| Service does not start | Inspect `data/service.log` and check for a port conflict or another worker sharing the data root. |
| Wait timed out | Read the same job with `status` or `wait`; the timeout does not cancel it. |
| Remote path cannot be read | Paths refer to the service computer; upload/download bytes or configure an explicit host connection. |

For startup debugging, stop the daemon and use `flux-image serve` in the foreground. Cancellation is cooperative at stage/denoising boundaries; it does not promise an immediate interruption of weight loading.

## Updating and removing the installation

```powershell
flux-image stop
git pull --ff-only
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

Re-register if you move the checkout or replace its environment. Existing conflicting entries need inspection; moving a folder does not fix absolute executable paths.

There is no automatic uninstaller. Stop the service, remove only its `local-flux` entries from registered harnesses, remove this checkout's `bin` directory from your user PATH, then remove the checkout after retaining any outputs you need. Hugging Face authentication and uv's Python/cache are separate user resources.

## Attribution and model licenses

The ConvRot loader is adapted from [animede/flux3-klein-studio](https://github.com/animede/flux3-klein-studio). Its Apache 2.0 notice is retained. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the [included upstream license](vendor/LICENSE.animede).

Model weights are separate downloads retaining their publishers' terms. This repository does not distribute them or replace their licenses. Review the [FLUX 3 Image Klein license](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md) for the intended use. Dependencies retain their respective licenses.
