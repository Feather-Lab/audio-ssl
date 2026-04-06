"""
Layerwise zero-shot evaluation of the pretrained AudioMAE encoder.

Evaluates AudioMAE (hance-ai/audiomae, ViT-Base) on three triplet tasks,
extracting time-pooled embeddings from every transformer block:
  1. mandarin       — Mandarin tone discrimination  (TonePerfectDataset)
  2. speech_commands — Speech commands level discrimination
  3. nsynth         — NSynth melody match  (NsynthTripletDataset)

SLURM array support
-------------------
Each task is one array element (AudioMAE has a single model size).

    # enumerate jobs, then submit
    python eval_audiomae_layerwise_zero_shot.py --list_jobs
    sbatch --array=0-2 eval_audiomae_layerwise_zero_shot.sh

    # or run a single job locally
    python eval_audiomae_layerwise_zero_shot.py --job_id 0

    # gather per-job CSVs into one file per task
    python eval_audiomae_layerwise_zero_shot.py --gather
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lightning_scripts"))

from lightning_scripts.audiomae_encoder_utils import (
    AUDIOMAE_SR,
    AudioMAELayerwiseEncoder,
)
from lightning_scripts.zero_shot_utils import embedding_triplet_metrics

TASK_NAMES = ["mandarin", "speech_commands", "nsynth"]


# ---------------------------------------------------------------------------
# Generic layerwise evaluation loop
# ---------------------------------------------------------------------------

def evaluate_layerwise(
    encoder: AudioMAELayerwiseEncoder,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str,
    extract_meta_fn: Callable,
    desc: str = "Evaluating",
) -> List[Dict]:
    records: List[Dict] = []
    for batch_idx, (clips, sr, triplet) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc=desc,
    ):
        if any(clips[k] is None for k in ("anchor", "positive", "negative")):
            continue

        emb_a = encoder(clips["anchor"].unsqueeze(0).unsqueeze(0).to(device))
        emb_p = encoder(clips["positive"].unsqueeze(0).unsqueeze(0).to(device))
        emb_n = encoder(clips["negative"].unsqueeze(0).unsqueeze(0).to(device))

        meta = extract_meta_fn(batch_idx, triplet)

        for layer_name in emb_a:
            metrics = embedding_triplet_metrics(
                emb_a[layer_name], emb_p[layer_name], emb_n[layer_name],
            )
            metrics.update(meta)
            metrics["model_name"] = model_name
            metrics["layer"] = layer_name
            records.append(metrics)

    return records


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _triplet_collate(batch):
    clips, sr, triplet = batch[0]
    return clips, sr, triplet


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_mandarin(encoder, model_name, device, n_examples=2000):
    from lightning_scripts.tone_perfect_triplet_dataset import TonePerfectDataset

    tone_perfect_dir = Path("~/ceph/datasets/tone_perfect").expanduser()
    ds = TonePerfectDataset(
        tone_perfect_dir,
        resample_sr=AUDIOMAE_SR,
        pair_mode=True,
        include_negative=True,
        allow_tone1=True,
        n_examples=n_examples,
        random_seed=0,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=_triplet_collate,
    )

    def extract_meta(batch_idx, triplet):
        return {
            "example_idx": batch_idx,
            "speaker": triplet["anchor"].get("speaker"),
            "tone": triplet["anchor"].get("tone"),
            "syllable_a": triplet["anchor"].get("syllable"),
            "syllable_b": triplet["positive"].get("syllable"),
            "neg_tone": triplet["negative"].get("tone"),
        }

    return evaluate_layerwise(
        encoder, loader, device, model_name, extract_meta,
        desc=f"Mandarin tone — {model_name}",
    )


def run_speech_commands(encoder, model_name, device, n_examples=2000):
    from datasets import load_dataset
    from lightning_scripts.speech_commands_triplet_dataset import (
        SpeechCommandsLevelTripletDataset,
    )

    print("  Loading Speech Commands dataset …")
    speech_commands = load_dataset(
        "google/speech_commands", "v0.01", trust_remote_code=True,
    )
    val_split = speech_commands["validation"]

    ds = SpeechCommandsLevelTripletDataset(
        val_split, seed=0, num_triplets=n_examples,
        target_sr=AUDIOMAE_SR, sig_length=int(AUDIOMAE_SR * 2),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=_triplet_collate,
    )

    def extract_meta(batch_idx, triplet):
        return {
            "example_idx": batch_idx,
            "word1": triplet["anchor"]["word1"],
            "word2": triplet["anchor"]["word2"],
            "word1_higher": triplet["anchor"]["word1_higher"],
        }

    return evaluate_layerwise(
        encoder, loader, device, model_name, extract_meta,
        desc=f"Speech Commands level — {model_name}",
    )


def run_nsynth(encoder, model_name, device, n_examples=5000):
    from lightning_scripts.nsynth_triplet_dataset import NsynthTripletDataset

    ds = NsynthTripletDataset(
        n_examples=n_examples,
        seed=42,
        target_sr=AUDIOMAE_SR,
        min_midi=30,
        max_midi=90,
        experiment_type="melody_match",
        balance_eval=True,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=_triplet_collate,
    )

    def extract_meta(batch_idx, triplet):
        interval = triplet["anchor"]["interval"]
        negative_interval = triplet["negative"]["interval"]
        return {
            "example_idx": batch_idx,
            "interval": interval,
            "instrument": triplet["anchor"]["instrument"],
            "negative_interval": negative_interval,
            "interval_diff": negative_interval - interval,
        }

    return evaluate_layerwise(
        encoder, loader, device, model_name, extract_meta,
        desc=f"NSynth melody match — {model_name}",
    )


TASK_RUNNERS = {
    "mandarin": run_mandarin,
    "speech_commands": run_speech_commands,
    "nsynth": run_nsynth,
}


# ---------------------------------------------------------------------------
# Job array helpers
# ---------------------------------------------------------------------------

def build_job_list():
    """Return list of task names — one per SLURM array element."""
    return list(TASK_NAMES)


def job_csv_path(out_dir: Path, task: str) -> Path:
    return out_dir / f"zero_shot_{task}_audiomae_layerwise.csv"


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------

def gather(out_dir: Path) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for task in TASK_NAMES:
        p = job_csv_path(out_dir, task)
        if p.exists():
            df = pd.read_csv(p)
            print(f"  read {p}  ({len(df)} rows)")
            result[task] = df
        else:
            print(f"  MISSING {p}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layerwise AudioMAE zero-shot evaluation",
    )
    parser.add_argument(
        "--job_id", type=int, default=None,
        help="Job index (0-2 for the three tasks). "
             "Falls back to SLURM_ARRAY_TASK_ID env var.",
    )
    parser.add_argument("--out_dir", type=str, default="results_dfs")
    parser.add_argument(
        "--n_examples", type=int, default=None,
        help="Override per-task example count",
    )
    parser.add_argument(
        "--list_jobs", action="store_true",
        help="Print the job array mapping and exit",
    )
    parser.add_argument(
        "--gather", action="store_true",
        help="Print status of result CSVs and exit",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_job_list()

    if args.list_jobs:
        for i, task in enumerate(jobs):
            print(f"  {i}: task={task}")
        print(f"\nTotal jobs: {len(jobs)}  (use --array=0-{len(jobs) - 1})")
        return

    if args.gather:
        gather(out_dir)
        return

    job_id = args.job_id
    if job_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is not None:
            job_id = int(env_id)
        else:
            parser.error("Provide --job_id or set SLURM_ARRAY_TASK_ID")

    task = jobs[job_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Job {job_id}: task={task}  model=AudioMAE")

    print("Loading AudioMAE …")
    encoder = AudioMAELayerwiseEncoder(time_pool=True).to(device)
    print(f"  {encoder.n_blocks} ViT blocks")

    runner = TASK_RUNNERS[task]
    kwargs = dict(encoder=encoder, model_name="AudioMAE", device=device)
    if args.n_examples is not None:
        kwargs["n_examples"] = args.n_examples

    records = runner(**kwargs)
    df = pd.DataFrame(records)

    csv_path = job_csv_path(out_dir, task)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows → {csv_path}")

    summary = (
        df.groupby("layer")
        [["cos_judgement", "r_judgement", "sqr_l2_judgement"]]
        .mean()
    )
    print(summary.to_string())

    print("\nGathering all available results …")
    gather(out_dir)


if __name__ == "__main__":
    main()
