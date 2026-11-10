from __future__ import annotations

import gc
import importlib.util
import json
import math

from PIL import Image

from .config import Settings, select_gpu
from .schema import ImageRequest


class FluxBackend:
    """One GPU-resident INT8 DiT + NF4 encoder. No offload hooks or CPU fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded = False
        self.pipes = []
        self.memory = {}

    def readiness(self) -> dict:
        required = {
            "pipeline_config": self.settings.models / "flux3/modular_model_index.json",
            "transformer": self.settings.models / "int8/diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors",
            "encoder": self.settings.models / "flux3/text_encoder/config.json",
            "vae": self.settings.models / "flux3/vae/config.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        encoder_root = self.settings.models / "flux3/text_encoder"
        index_path = encoder_root / "model.safetensors.index.json"
        if index_path.is_file():
            shards = set(json.loads(index_path.read_text("utf-8")).get("weight_map", {}).values())
            if not shards or any(not (encoder_root / name).is_file() for name in shards):
                missing.append("encoder_weights")
        elif not (encoder_root / "model.safetensors").is_file():
            missing.append("encoder_weights")
        if not (self.settings.models / "flux3/vae/diffusion_pytorch_model.safetensors").is_file():
            missing.append("vae_weights")
        packages = [name for name in ("torch", "diffusers", "transformers", "bitsandbytes", "comfy_kitchen")
                    if importlib.util.find_spec(name) is None]
        downloads = []
        if self.settings.models.exists():
            for resume in self.settings.models.rglob("*.resume.json"):
                try:
                    state = json.loads(resume.read_text("utf-8"))
                    size = state["identity"]["size"]
                    downloads.append({"file": state["identity"]["file"],
                                      "ranges_complete": len(state["complete"]),
                                      "ranges_total": math.ceil(size / (16 * 1024 * 1024))})
                except (OSError, ValueError, KeyError):
                    pass
        return {"loaded": self.loaded, "ready": not missing and not packages and bool(self.settings.gpu_uuid),
                "missing_files": missing, "missing_packages": packages,
                "gpu_uuid": self.settings.gpu_uuid, "memory_budget_gib": self.settings.memory_gib,
                "model": "FLUX 3 Image Klein distilled INT8 ConvRot", "text_encoder": "Qwen3-VL-4B NF4",
                "upsampler": False, "cpu_offload": False, "last_memory": self.memory, "downloads": downloads}

    def _load(self, progress):
        ready = self.readiness()
        if ready["missing_files"] or ready["missing_packages"]:
            raise RuntimeError(f"Inference not ready: {ready}. Run flux-image setup-models and install inference dependencies.")
        select_gpu(self.settings)
        import torch
        from diffusers import ModularPipeline
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )

        from ._convrot import load_flux3_convrot_transformer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly the configured CUDA GPU must be visible")
        if "RTX 3060" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Refusing inference on a GPU other than the configured RTX 3060")
        total = torch.cuda.get_device_properties(0).total_memory
        budget = min(self.settings.memory_gib * 1024**3, total - 1536 * 1024**2)
        torch.cuda.set_per_process_memory_fraction(budget / total, 0)
        # Keep allocations below dedicated VRAM capacity instead of relying on WDDM shared memory.
        root = self.settings.models / "flux3"
        modular = ModularPipeline.from_pretrained(root, local_files_only=True)
        blocks = modular.blocks.sub_blocks
        if "prompt_upsampler" in blocks:
            blocks.pop("prompt_upsampler")
        dtype = torch.bfloat16
        progress("loading_text_encoder", 2)
        text = blocks["text_encoder"].init_pipeline(root)
        self.pipes.append(text)
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            root / "text_encoder", local_files_only=True, dtype=dtype,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
            device_map={"": 0},
        )
        processor = AutoProcessor.from_pretrained(root / "text_encoder", local_files_only=True)
        text.register_components(text_encoder=encoder, tokenizer=processor)
        text.load_components(local_files_only=True)
        # Do not cast the already-quantized encoder or its quantization state.
        progress("loading_transformer", 5)
        transformer = load_flux3_convrot_transformer(
            self.settings.models / "int8/diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors",
            root / "transformer", compute_dtype=dtype,
        ).to("cuda:0")
        denoise = blocks["denoise"].init_pipeline(root)
        self.pipes.append(denoise)
        denoise.register_components(transformer=transformer)
        denoise.load_components(local_files_only=True)
        denoise.to("cuda:0")
        vae_encode = blocks["vae_encoder"].init_pipeline(root)
        self.pipes.append(vae_encode)
        vae_encode.load_components(dtype=dtype, local_files_only=True)
        vae_encode.to("cuda:0")
        decode = blocks["decode"].init_pipeline(root)
        self.pipes.append(decode)
        decode.register_components(vae=vae_encode.vae)
        decode.load_components(dtype=dtype, local_files_only=True)
        decode.to("cuda:0")
        vae_encode.vae.enable_tiling()
        # Smaller GPU-only tiles reserve space for transformer/encoder residency.
        vae_encode.vae.tile_sample_min_size = 256
        vae_encode.vae.tile_latent_min_size = 256 // (2 ** (len(vae_encode.vae.config.block_out_channels) - 1))
        self.text, self.denoise, self.vae_encode, self.decode = text, denoise, vae_encode, decode
        self.audit_gpu()
        self.loaded = True

    def audit_gpu(self):
        wrong = []
        resident_bytes = 0
        seen = set()
        for name, module in (("text_encoder", self.text.text_encoder), ("transformer", self.denoise.transformer), ("vae", self.decode.vae)):
            if getattr(module, "hf_device_map", None):
                if any(str(d) in {"cpu", "disk"} for d in module.hf_device_map.values()):
                    raise RuntimeError(f"{name} has a CPU/disk device mapping")
            for n, m in module.named_modules():
                if getattr(m, "_hf_hook", None) is not None and getattr(m._hf_hook, "offload", False):
                    raise RuntimeError(f"Offloading hook detected: {name}.{n}")
            for n, tensor in list(module.named_parameters()) + list(module.named_buffers()):
                if tensor.device.type != "cuda":
                    wrong.append(f"{name}.{n}: {tensor.device}")
                if tensor.data_ptr() not in seen:
                    resident_bytes += tensor.numel() * tensor.element_size()
                    seen.add(tensor.data_ptr())
        if wrong:
            raise RuntimeError("GPU residency audit failed: " + "; ".join(wrong[:12]))
        return {"all_model_tensors_on_cuda": True, "resident_model_mib": round(resident_bytes / 1024**2, 1),
                "offload_hooks": False}

    def unload(self):
        self.loaded = False
        self.pipes.clear()
        for name in ("text", "denoise", "vae_encode", "decode"):
            self.__dict__.pop(name, None)
        gc.collect()
        if "torch" in __import__("sys").modules:
            import torch
            torch.cuda.empty_cache()

    def generate(self, req: ImageRequest, job_id: str, progress, cancelled) -> dict:
        from .metrics import GPUMemorySampler
        sampler = GPUMemorySampler().start()
        try:
            return self._generate(req, job_id, progress, cancelled, sampler)
        except BaseException:
            self.unload()
            raise
        finally:
            sampler.stop()

    def _generate(self, req, job_id, progress, cancelled, sampler):
        progress("loading", 1)
        if not self.loaded:
            self._load(progress)
        import torch
        torch.cuda.reset_peak_memory_stats()
        images = []
        # Bound total reference tokens independently of output size.
        per_image = 1024 * 1024 // max(1, len(req.reference_images))
        for asset_id in req.reference_images:
            with Image.open(self.settings.home / "inputs" / f"{asset_id}.png") as src:
                image = src.convert("RGB")
                scale = min(1, math.sqrt(per_image / (image.width * image.height)))
                width, height = max(16, int(image.width * scale) // 16 * 16), max(16, int(image.height * scale) // 16 * 16)
                images.append(image.resize((width, height), Image.Resampling.LANCZOS))
        def move_state(state):
            for key, value in list(state.values.items()):
                if isinstance(value, torch.Tensor):
                    state.values[key] = value.to("cuda:0")
            return state
        step = 0
        def before_step(module, args):
            nonlocal step
            step += 1
            progress("denoising", 20 + int(65 * (step - 1) / req.steps))
        hook = self.denoise.transformer.register_forward_pre_hook(before_step)
        try:
            with torch.inference_mode():
                progress("encoding_prompt", 10)
                tokens = self.text.tokenizer.tokenizer(req.prompt, add_special_tokens=False)["input_ids"]
                if len(tokens) > 1024:
                    raise ValueError("Prompt exceeds the 1024-token GPU profile limit; shorten the scene caption")
                state = move_state(self.text(prompt=req.prompt, max_sequence_length=1120))
                if images:
                    progress("encoding_references", 15)
                    state = self.vae_encode(state=state, image=images)
                self.denoise.scheduler.set_shift(16.0)
                state = self.denoise(state=state, width=req.width, height=req.height,
                    num_inference_steps=req.steps, generator=torch.Generator(device="cuda:0").manual_seed(req.seed))
                progress("decoding", 90)
                image = self.decode(state=state, output="images", output_type="pil")[0]
                torch.cuda.synchronize()
        finally:
            hook.remove()
        progress("saving", 97)
        residency = self.audit_gpu()
        self.memory = {"peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                       "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
                       "gpu_name": torch.cuda.get_device_name(0), "gpu_uuid": self.settings.gpu_uuid,
                       "residency": residency, "windows_driver": sampler.stop()}
        path = self.settings.home / "outputs" / f"{job_id}.png"
        from PIL.PngImagePlugin import PngInfo
        metadata = {"request": req.model_dump(), "memory": self.memory,
                    "model": "FLUX 3 Image Klein distilled INT8 ConvRot", "text_encoder": "NF4", "upsampler": False}
        pnginfo = PngInfo()
        pnginfo.add_text("flux_image", json.dumps(metadata))
        image.save(path, pnginfo=pnginfo)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"image_path": str(path), "image_url": f"/v1/jobs/{job_id}/image", "seed": req.seed,
                "width": image.width, "height": image.height, "memory": self.memory}
