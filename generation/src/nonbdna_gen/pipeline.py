import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .evo2_apptainer import run_evo2_window
from .manifest import write_manifest_csv
from .provenance import collect_provenance
from .windowing import sample_windows_and_write

def _read_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _detect_repo_root(fallback_cwd: Path) -> Path:
    # best-effort: works for users who cloned the repo
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=fallback_cwd, text=True).strip()
        if out:
            return Path(out)
    except Exception:
        pass
    return fallback_cwd.resolve()

def _decompress_ref(ref_gz: Path, ref_fa: Path, force: bool) -> None:
    if ref_fa.exists() and ref_fa.stat().st_size > 0 and not force:
        return
    _ensure_dir(ref_fa.parent)
    if str(ref_gz).endswith(".gz"):
        try:
            subprocess.run(["bash", "-lc", f"zcat -- '{ref_gz}' > '{ref_fa}'"], check=True)
        except Exception:
            import gzip
            with gzip.open(ref_gz, "rt") as fin, open(ref_fa, "w") as fout:
                shutil.copyfileobj(fin, fout)
    else:
        shutil.copyfile(ref_gz, ref_fa)

def run_pipeline(*, config_path: Path, workdir: Path) -> None:
    cfg = _read_yaml(config_path)

    run = cfg.get("run", {})
    paths = cfg.get("paths", {})
    windows_cfg = cfg.get("windows", {})
    gen_cfg = cfg.get("generation", {})
    appt_cfg = cfg.get("apptainer", {})

    genome_tag = run["genome_tag"]
    seed = int(run.get("seed", 1337))
    force = bool(run.get("force", False))

    ref_gz = Path(paths["ref_fasta_gz"])
    outdir = Path(paths["outdir"])
    logroot = Path(paths.get("logroot", str(workdir / "logs")))

    _ensure_dir(workdir)
    _ensure_dir(outdir)
    _ensure_dir(logroot)

    # Artifacts inside workdir
    ref_fa = workdir / "ref.fa"
    win_dir = workdir / "windows"
    win_tsv = workdir / "windows.tsv"
    meta_json = workdir / f"run_meta.{genome_tag}.json"
    prov_json = workdir / "provenance.json"
    manifest_csv = workdir / f"pairs.{genome_tag}.csv"
    resolved_cfg_path = workdir / "config.resolved.yaml"

    _ensure_dir(win_dir)

    # Save config snapshot
    with open(resolved_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Provenance snapshot
    prov = collect_provenance()
    with open(prov_json, "w") as f:
        json.dump(prov, f, indent=2)

    # Reference
    _decompress_ref(ref_gz, ref_fa, force=force)

    # Sample windows + extract originals
    sampled: List[Tuple[str, int, int]] = sample_windows_and_write(
        ref_fa=ref_fa,
        n_windows=int(windows_cfg.get("n_windows", 100)),
        window_len=int(windows_cfg.get("window_len", 1_000_000)),
        min_gap=int(windows_cfg.get("min_gap", 100_000)),
        n_threshold=float(windows_cfg.get("n_threshold", 0.10)),
        seed=seed,
        primary_only=bool(windows_cfg.get("primary_contigs_only", True)),
        win_tsv=win_tsv,
        meta_json=meta_json,
        win_dir=win_dir,
    )

    # Generation settings
    n_gpus = int(gen_cfg.get("n_gpus", 1))
    gpu_ids = list(gen_cfg.get("gpu_ids", list(range(n_gpus))))
    if len(gpu_ids) < n_gpus:
        raise ValueError(f"gpu_ids length {len(gpu_ids)} < n_gpus {n_gpus}")

    model_name = str(gen_cfg.get("model_name", "evo2_7b"))
    temperature = float(gen_cfg.get("temperature", 1.0))
    top_k = int(gen_cfg.get("top_k", 4))
    disable_gbif = bool(gen_cfg.get("disable_gbif", False))
    phylo_tag = str(gen_cfg.get("phylo_tag", "")).strip()

    # Apptainer settings
    sif = str(appt_cfg["sif"])
    hf_cache = str(appt_cfg["hf_cache"])
    binds = list(appt_cfg.get("binds", []))
    envmap = dict(appt_cfg.get("env", {}))

    repo_root_cfg = appt_cfg.get("repo_root", "auto")
    if isinstance(repo_root_cfg, str) and repo_root_cfg.lower() == "auto":
        repo_root = _detect_repo_root(config_path.parent)
    else:
        repo_root = Path(str(repo_root_cfg)).resolve()

    # Build jobs
    jobs = []
    for i, (contig, start0, length) in enumerate(sampled, start=1):
        orig_fa = win_dir / f"orig.{contig}.{start0}.{length}.fa"
        end0 = start0 + length - 1
        syn_base = f"syn_{contig}:{start0+1}-{end0}"
        syn_fa = outdir / f"{syn_base}.fasta"
        syn_meta = outdir / f"{syn_base}.json"
        jobs.append((i, contig, start0, length, orig_fa, syn_fa, syn_meta))

    def _one(pair_i: int, gpu: int, orig_fa: Path, syn_fa: Path, syn_meta: Path) -> None:
        if syn_fa.exists() and syn_fa.stat().st_size > 0 and not force:
            return
        run_evo2_window(
            sif=sif,
            hf_cache=hf_cache,
            repo_root=str(repo_root),
            binds=binds,
            envmap=envmap,
            cuda_visible_devices=str(gpu),
            model_name=model_name,
            temperature=temperature,
            top_k=top_k,
            disable_gbif=disable_gbif,
            phylo_tag=phylo_tag,
            prompt_fasta=str(orig_fa),
            out_fasta=str(syn_fa),
            out_meta=str(syn_meta),
            logdir=str(logroot),
        )
        if not (syn_fa.exists() and syn_fa.stat().st_size > 0):
            raise RuntimeError(f"Expected synthetic output not found: {syn_fa}")

    # Run parallel
    futures = []
    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        for j, (pair_i, contig, start0, length, orig_fa, syn_fa, syn_meta) in enumerate(jobs):
            gpu = gpu_ids[j % n_gpus]
            futures.append(ex.submit(_one, pair_i, gpu, orig_fa, syn_fa, syn_meta))
        for fut in as_completed(futures):
            fut.result()

    # Write manifest
    rows = []
    for (pair_i, contig, start0, length, orig_fa, syn_fa, _syn_meta) in jobs:
        rows.append({
            "id": f"pair_{pair_i:03d}",
            "orig": str(orig_fa),
            "syn": str(syn_fa),
            "contig": contig,
            "start": str(start0),
            "len": str(length),
            "seed": str(seed),
        })
    write_manifest_csv(manifest_csv, rows)

    print(f"[done] manifest: {manifest_csv}")
    print(f"[done] outdir:   {outdir}")
    print(f"[done] workdir:  {workdir}")
