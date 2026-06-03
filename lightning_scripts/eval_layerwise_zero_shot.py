"""
Unified layerwise zero-shot evaluation for CochCNN9, Whisper, and AudioMAE.

Evaluates models on three triplet tasks, extracting embeddings from every
internal layer in a single forward pass:
  1. mandarin       — Mandarin tone discrimination  (TonePerfectDataset)
  2. speech_commands — Speech commands level discrimination
  3. nsynth         — NSynth melody match  (NsynthTripletDataset)

Each (task, model) pair is one SLURM array element.

Usage
-----
    # List all jobs for a model type
    python eval_layerwise_zero_shot.py --model_type cochdnn --list_jobs
    python eval_layerwise_zero_shot.py --model_type whisper --list_jobs
    python eval_layerwise_zero_shot.py --model_type audiomae --list_jobs

    # Run a single job
    python eval_layerwise_zero_shot.py --model_type cochdnn --job_id 0

    # Gather per-job CSVs into merged files
    python eval_layerwise_zero_shot.py --model_type cochdnn --gather
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import torch
from tqdm.auto import tqdm

from default_paths import TONE_PERFECT_DIR, require_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lightning_scripts"))

from lightning_scripts.zero_shot_utils import (
    COCHDNN_MODEL_REGISTRY,
    MODEL_SR,
    embedding_triplet_metrics,
    load_single_cochdnn_model,
)

TASK_NAMES = ["mandarin", "speech_commands", "nsynth"]

WHISPER_SR = 16_000
WHISPER_MODELS = {"tiny": "tiny", "medium": "medium", "large-v3": "large-v3"}

AUDIOMAE_SR = 16_000


# ---------------------------------------------------------------------------
# Generic layerwise evaluation loop (shared across all model types)
# ---------------------------------------------------------------------------

def evaluate_layerwise(
    encoder: Callable,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str,
    extract_meta_fn: Callable,
    desc: str = "Evaluating",
) -> List[Dict]:
    """Run *encoder* on every triplet in *dataloader*, returning per-layer metrics."""
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


def _triplet_collate(batch):
    clips, sr, triplet = batch[0]
    return clips, sr, triplet


# ---------------------------------------------------------------------------
# Task runners (parameterised by target_sr)
# ---------------------------------------------------------------------------

def run_mandarin(encoder, model_name, device, target_sr, n_examples=2000):
    from lightning_scripts.tone_perfect_triplet_dataset import TonePerfectDataset

    tone_perfect_dir = require_path(
        TONE_PERFECT_DIR,
        "COCHDNN_TONE_PERFECT_DIR",
        "Tone Perfect dataset",
    )
    ds = TonePerfectDataset(
        tone_perfect_dir,
        resample_sr=target_sr,
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


def run_speech_commands(encoder, model_name, device, target_sr, n_examples=2000):
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
        target_sr=target_sr, sig_length=int(target_sr * 2),
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


def run_nsynth(encoder, model_name, device, target_sr, n_examples=5000):
    from lightning_scripts.nsynth_triplet_dataset import NsynthTripletDataset

    ds = NsynthTripletDataset(
        n_examples=n_examples,
        seed=42,
        target_sr=target_sr,
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
# Encoder construction per model_type
# ---------------------------------------------------------------------------

def build_encoder(model_type: str, model_key: str, device: torch.device):
    """Return ``(encoder_callable, display_name, target_sr, n_layers_info)``."""

    if model_type == "cochdnn":
        encoder, display_name, layer_names = load_single_cochdnn_model(
            model_key, device=str(device),
        )
        return encoder, display_name, MODEL_SR, f"{len(layer_names)} kell2018 layers"

    if model_type == "whisper":
        from lightning_scripts.eval_whisper_layerwise_zero_shot import (
            WhisperLayerwiseEncoder,
        )
        whisper_name = WHISPER_MODELS[model_key]
        encoder = WhisperLayerwiseEncoder(whisper_name).to(device)
        return (
            encoder,
            f"Whisper {whisper_name}",
            WHISPER_SR,
            f"{encoder.n_blocks} encoder blocks",
        )

    if model_type == "audiomae":
        from lightning_scripts.audiomae_encoder_utils import (
            AudioMAELayerwiseEncoder,
        )
        encoder = AudioMAELayerwiseEncoder(time_pool=True).to(device)
        return encoder, "AudioMAE", AUDIOMAE_SR, f"{encoder.n_blocks} ViT blocks"

    raise ValueError(f"Unknown model_type: {model_type!r}")


# ---------------------------------------------------------------------------
# Job grid & CSV paths
# ---------------------------------------------------------------------------

def _model_keys_for_type(model_type: str) -> List[str]:
    if model_type == "cochdnn":
        return list(COCHDNN_MODEL_REGISTRY.keys())
    if model_type == "whisper":
        return list(WHISPER_MODELS.keys())
    if model_type == "audiomae":
        return ["audiomae"]
    raise ValueError(f"Unknown model_type: {model_type!r}")


def build_job_list(model_type: str) -> List[tuple]:
    """Return ``[(task, model_key), ...]`` — one per SLURM array element."""
    model_keys = _model_keys_for_type(model_type)
    jobs = []
    for task in TASK_NAMES:
        for mk in model_keys:
            jobs.append((task, mk))
    return jobs


def job_csv_path(out_dir: Path, model_type: str, task: str, model_key: str) -> Path:
    return out_dir / f"zero_shot_{task}_{model_type}_{model_key}_layerwise.csv"


def gather(out_dir: Path, model_type: str) -> Dict[str, pd.DataFrame]:
    """Concatenate per-job CSVs into one DataFrame per task and write merged CSVs."""
    model_keys = _model_keys_for_type(model_type)
    result: Dict[str, pd.DataFrame] = {}
    for task in TASK_NAMES:
        parts = []
        for mk in model_keys:
            p = job_csv_path(out_dir, model_type, task, mk)
            if p.exists():
                parts.append(pd.read_csv(p))
                print(f"  read {p}  ({len(parts[-1])} rows)")
            else:
                print(f"  MISSING {p}")
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged_path = out_dir / f"zero_shot_{task}_{model_type}_layerwise.csv"
            merged.to_csv(merged_path, index=False)
            print(f"  → {merged_path}  ({len(merged)} rows)")
            result[task] = merged
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified layerwise zero-shot evaluation",
    )
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["cochdnn", "whisper", "audiomae"],
        help="Model family to evaluate",
    )
    parser.add_argument(
        "--job_id", type=int, default=None,
        help="Job index into the (task, model) grid. "
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
        help="Concatenate per-job CSVs into merged files and exit",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_job_list(args.model_type)

    if args.list_jobs:
        for i, (task, model_key) in enumerate(jobs):
            print(f"  {i}: task={task}  model={model_key}")
        print(f"\nTotal jobs: {len(jobs)}  (use --array=0-{len(jobs) - 1})")
        return

    if args.gather:
        gather(out_dir, args.model_type)
        return

    # Resolve job_id
    job_id = args.job_id
    if job_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is not None:
            job_id = int(env_id)
        else:
            parser.error("Provide --job_id or set SLURM_ARRAY_TASK_ID")

    task, model_key = jobs[job_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Job {job_id}: model_type={args.model_type}  task={task}  model={model_key}")

    encoder, display_name, target_sr, info_str = build_encoder(
        args.model_type, model_key, device,
    )
    print(f"Loaded {display_name}  ({info_str})")

    runner = TASK_RUNNERS[task]
    kwargs = dict(
        encoder=encoder, model_name=display_name,
        device=device, target_sr=target_sr,
    )
    if args.n_examples is not None:
        kwargs["n_examples"] = args.n_examples

    records = runner(**kwargs)
    df = pd.DataFrame(records)

    csv_path = job_csv_path(out_dir, args.model_type, task, model_key)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows → {csv_path}")

    summary = (
        df.groupby("layer")
        [["cos_judgement", "r_judgement", "sqr_l2_judgement"]]
        .mean()
    )
    print(summary.to_string())

    print("\nGathering all available results …")
    gather(out_dir, args.model_type)


if __name__ == "__main__":
    main()
