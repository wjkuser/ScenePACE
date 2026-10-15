"""Start one configured local service in the foreground."""

import argparse, os
from urllib.parse import urlparse
from scene_pace.runtime import ROOT, path, load, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("service", choices=["actor", "vlm", "trellis", "isaac"])
    p.add_argument("--config", default="configs/runtime.json")
    a = p.parse_args()
    c = load(a.config)
    e = environment(c)
    name = a.service
    model = path(c["paths"]["models"])
    e.pop("PYTHONHOME", None)
    if name in ["actor", "vlm"]:
        o = c[name]
        if name == "actor" and o["mode"] != "local":
            raise ValueError("API actors do not require a local vLLM service")
        e["CUDA_VISIBLE_DEVICES"] = o["gpus"]
        e.pop("PYTHONPATH", None)
        cmd = [
            str(path(c["python"]["vllm"])),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model / o["model"]),
            "--served-model-name",
            o["served_name"],
            "--host",
            o.get("bind_host", "127.0.0.1"),
            "--port",
            str(urlparse(o["endpoint"]).port),
            "--tensor-parallel-size",
            str(o["tensor_parallel_size"]),
            "--max-model-len",
            str(o["context_tokens"]),
            "--gpu-memory-utilization",
            "0.85",
            "--max-num-seqs",
            "2",
            "--api-key",
            e.get(o["api_key_env"], "EMPTY"),
        ]
        if name == "actor":
            cmd += [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--reasoning-parser",
                "qwen3",
            ]
        else:
            cmd += [
                "--mm-processor-cache-gb",
                "0",
                "--reasoning-parser",
                "qwen3",
                "--gdn-prefill-backend",
                "triton",
            ]
    elif name == "trellis":
        e["CUDA_VISIBLE_DEVICES"] = c["trellis"]["gpus"]
        e["PYTHONPATH"] = str(path(c["trellis"]["source"]))
        cmd = [
            str(path(c["python"]["trellis"])),
            str(ROOT / "server/trellis_local_server.py"),
            "--port",
            str(urlparse(c["trellis"]["endpoint"]).port),
        ]
    else:
        e.pop("CUDA_VISIBLE_DEVICES", None)
        e.pop("PYTHONPATH", None)
        e["OMNI_KIT_ACCEPT_EULA"] = "Y"
        cmd = [str(path(c["python"]["isaac"])), str(ROOT / "scripts/start_isaac.py")]
    os.chdir(ROOT)
    os.execvpe(cmd[0], cmd, e)


if __name__ == "__main__":
    main()
