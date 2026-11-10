from __future__ import annotations

import json
import ssl

from .config import Settings

BASE_REPO = "black-forest-labs/flux-3-image-klein"
BASE_REV = "337337621b8526e120f2275d250242f4a6bedd75"
INT8_REPO = "black-forest-labs/flux-3-image-klein-comfy"
INT8_REV = "5c9b714851dc072b04a23ef6d2d36421f5ccb3a0"
INT8_FILE = "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors"


def setup_models(settings: Settings):
    """Use existing HF authentication, without copying credentials into this project."""
    import os
    # Standard HTTPS supports the system certificate store and resumable local downloads.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import httpx
    import truststore
    from huggingface_hub import set_client_factory, snapshot_download
    from huggingface_hub.errors import GatedRepoError
    set_client_factory(lambda: httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                                            timeout=httpx.Timeout(60, read=300)))
    root = settings.models
    try:
        snapshot_download(BASE_REPO, revision=BASE_REV, local_dir=root / "flux3", max_workers=2,
            allow_patterns=["modular_model_index.json", "transformer/config.json", "text_encoder/*", "vae/*",
                            "README.md", "LICENSE.md", "NOTICE.md"], ignore_patterns=["*.safetensors"],
        )
        from .download import verified_download
        index = json.loads((root / "flux3/text_encoder/model.safetensors.index.json").read_text("utf-8"))
        for shard in sorted(set(index["weight_map"].values())):
            verified_download(BASE_REPO, "text_encoder/" + shard, BASE_REV, root / "flux3/text_encoder" / shard)
        verified_download(BASE_REPO, "vae/diffusion_pytorch_model.safetensors", BASE_REV,
                          root / "flux3/vae/diffusion_pytorch_model.safetensors")
        verified_download(INT8_REPO, INT8_FILE, INT8_REV, root / "int8" / INT8_FILE)
    except GatedRepoError as exc:
        raise RuntimeError("Hugging Face model access is pending. Accept access for "
                           "https://huggingface.co/black-forest-labs/flux-3-image-klein and "
                           "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy "
                           "using the account behind the local HF token, then rerun flux-image setup-models.") from exc
    manifest = {"base": {"repo": BASE_REPO, "revision": BASE_REV},
                "transformer": {"repo": INT8_REPO, "revision": INT8_REV, "file": INT8_FILE},
                "upsampler": False}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
