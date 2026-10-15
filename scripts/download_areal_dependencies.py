"""Export the pinned AReaL public core dependencies, excluding optional integrations."""

import pathlib, tomllib

root = pathlib.Path(__file__).resolve().parents[1]
data = tomllib.loads((root / "third_party/AReaL/pyproject.vllm.toml").read_text())
# Optional agent frameworks and platform-specific launchers are not used by this project.
exclude = {
    "awex",
    "openhands",
    "langchain",
    "langchain-openai",
    "claude-agent-sdk",
    "qwen_agent",
    "swanboard",
    "swanlab",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "ray",
    "pillow",
    "omegaconf",
    "hydra-core",
}
import re

for dep in data["project"]["dependencies"]:
    name = re.split(r"[<>=!~;\[ ]", dep, maxsplit=1)[0].lower()
    if name not in exclude:
        print(dep)
