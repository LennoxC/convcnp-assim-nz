#!/usr/bin/env python3
"""
Execute a notebook in a torchrun-launched process.

This script is intended to be launched under `torch.distributed.run` (torchrun).
Each process will:
 - set `CUDA_DEVICE` from LOCAL_RANK
 - set `NOTEBOOK_LOG_FILE` to a per-rank timestamped logfile
 - execute the notebook using nbconvert's ExecutePreprocessor
 - write an executed notebook to `executed/<notebook>_executed_rank<N>.ipynb`

Usage example (from project root):
  torchrun --nproc_per_node=4 scripts/torchrun_nb_exec.py --notebook notebooks/experiments/exp2_transfer_learning.ipynb

"""
import os
import argparse
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True, help="Path to notebook to execute")
    args = parser.parse_args()

    nb_path = Path(args.notebook).resolve()
    if not nb_path.exists():
        raise SystemExit(f"Notebook not found: {nb_path}")

    # torchrun sets these env vars
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("CUDA_DEVICE", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # set CUDA_DEVICE for notebook compatibility
    os.environ["CUDA_DEVICE"] = str(local_rank)

    # set per-rank NOTEBOOK_LOG_FILE so notebook's setup_logging writes to it
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_file = logs_dir / f"{nb_path.stem}_{ts}_rank{rank}.log"
    os.environ["NOTEBOOK_LOG_FILE"] = str(log_file)

    print(f"[rank {rank}/{world_size}] executing notebook {nb_path} on local_rank={local_rank}")
    print(f"[rank {rank}] NOTEBOOK_LOG_FILE={os.environ['NOTEBOOK_LOG_FILE']}")

    # load notebook
    with nb_path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    ep = ExecutePreprocessor(timeout=-1, kernel_name="python3")

    out_dir = nb_path.parent / "executed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb = out_dir / f"{nb_path.stem}_executed_rank{rank}.ipynb"

    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
    except Exception as e:
        print(f"[rank {rank}] Notebook execution error: {e}")

    with out_nb.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)

    print(f"[rank {rank}] wrote executed notebook: {out_nb}")


if __name__ == "__main__":
    main()
