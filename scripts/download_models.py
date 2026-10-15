"""Download public model files; HF_ENDPOINT and HF_TOKEN are supported."""

import argparse, json, hashlib, urllib.request, os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--directory", default="models")
p.add_argument("--manifest", default="configs/models.json")
a = p.parse_args()
out = Path(a.directory).resolve()
out.mkdir(parents=True, exist_ok=True)
resolved = []
for item in json.loads((ROOT / a.manifest).read_text()):
    target = out / item["directory"]
    snapshot_download(
        item["repo"],
        revision=item["revision"],
        local_dir=target,
        allow_patterns=item.get("files"),
    )
    resolved.append(item)
matfuse = out / "MatFuse/checkpoints/matfuse-full.ckpt"
target = out / "matfuse-full.ckpt"
if not matfuse.is_file():
    raise FileNotFoundError(matfuse)
if not target.exists():
    try:
        os.link(matfuse, target)
    except OSError:
        shutil.copy2(matfuse, target)
clip_hash = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
clip = out / "ViT-B-16.pt"
if not clip.exists():
    urllib.request.urlretrieve(
        "https://openaipublic.azureedge.net/clip/models/" + clip_hash + "/ViT-B-16.pt",
        clip,
    )
if hashlib.sha256(clip.read_bytes()).hexdigest() != clip_hash:
    raise RuntimeError("CLIP checksum mismatch")
# TRELLIS references the image decoder and text encoder in its upstream JSON.
pipeline = out / "TRELLIS-text-xlarge/pipeline.json"
data = json.loads(pipeline.read_text())


def localize(value):
    if isinstance(value, dict):
        return {k: localize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [localize(v) for v in value]
    if isinstance(value, str):
        for repo, folder in [
            ("microsoft/TRELLIS-image-large", "TRELLIS-image-large"),
            ("openai/clip-vit-large-patch14", "clip-vit-large-patch14"),
        ]:
            if value == repo or value.startswith(repo + "/"):
                return str(out / folder) + value[len(repo) :]
    return value


pipeline.write_text(json.dumps(localize(data), indent=2))
(out / "download-manifest.json").write_text(json.dumps(resolved, indent=2))
