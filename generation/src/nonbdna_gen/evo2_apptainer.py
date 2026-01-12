import os
import subprocess
from typing import Dict, List

def run_evo2_window(
    *,
    sif: str,
    hf_cache: str,
    repo_root: str,
    binds: List[str],
    envmap: Dict[str, str],
    cuda_visible_devices: str,
    model_name: str,
    temperature: float,
    top_k: int,
    disable_gbif: bool,
    phylo_tag: str,
    prompt_fasta: str,
    out_fasta: str,
    out_meta: str,
    logdir: str,
) -> None:
    # Bind repo so container can run the generator shipped in GitHub
    bind_args = ["--bind", f"{repo_root}:/work/repo"]

    # Bind HF cache
    bind_args += ["--bind", f"{hf_cache}:/root/.cache/huggingface"]

    # Extra binds (scratch etc.)
    for b in binds:
        bind_args += ["--bind", b]

    # Container env
    env_vars = dict(envmap)
    env_vars["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        env_vars["HUGGING_FACE_HUB_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    env_joined = ",".join([f"{k}={v}" for k, v in env_vars.items()])

    container_script = "/work/repo/generation/evo2_generate_byseq.py"

    cmd = [
        "apptainer", "exec",
        "--nv",
        "--cleanenv",
        "--env", env_joined,
        *bind_args,
        sif,
        "python3", container_script,
        "--model", model_name,
        "--input", prompt_fasta,
        "--out_fasta", out_fasta,
        "--out_meta", out_meta,
        "--logdir", logdir,
        "--device", "cuda",
        "--temperature", str(temperature),
        "--top_k", str(top_k),
    ]
    if disable_gbif:
        cmd.append("--disable_gbif")
    if phylo_tag:
        cmd += ["--phylo-tag", phylo_tag]

    subprocess.run(cmd, check=True)
