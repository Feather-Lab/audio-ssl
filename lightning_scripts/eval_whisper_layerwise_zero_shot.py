"""
Layerwise zero-shot evaluation of pre-trained Whisper encoders.

Evaluates whisper-tiny and whisper-large-v3 on three triplet tasks,
extracting time-averaged embeddings from every encoder block:
  1. mandarin       — Mandarin tone discrimination  (TonePerfectDataset)
  2. speech_commands — Speech commands level discrimination
  3. nsynth         — NSynth melody match  (NsynthTripletDataset)

SLURM array support
-------------------
Each (task, model) pair is one array element.  All encoder layers are
evaluated in a single forward pass per triplet, so splitting by layer
would only duplicate model-loading cost.

    # enumerate jobs, then submit
    python eval_whisper_layerwise_zero_shot.py --list_jobs
    sbatch --array=0-5 eval_whisper_layerwise_zero_shot.sh

    # or run a single job locally
    python eval_whisper_layerwise_zero_shot.py --job_id 0

    # gather per-job CSVs into one file per task
    python eval_whisper_layerwise_zero_shot.py --gather
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import whisper
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lightning_scripts"))

from lightning_scripts.zero_shot_utils import embedding_triplet_metrics

WHISPER_SR = 16_000

WHISPER_MODELS = {
    "tiny": "tiny",
    "medium": "medium",
    "large-v3": "large-v3",
}

TASK_NAMES = ["mandarin", "speech_commands", "nsynth"]


# ---------------------------------------------------------------------------
# Whisper encoder with hooks on every block
# ---------------------------------------------------------------------------

class WhisperLayerwiseEncoder(nn.Module):
    """Wrap a pre-trained Whisper encoder; forward returns per-layer embeddings.

    Accepts raw 16 kHz waveform ``(B, 1, T)`` and returns
    ``Dict[layer_name, (B, D)]`` with time-averaged representations for
    every transformer block and the final ``ln_post`` output.
    """

    def __init__(self, whisper_model_name: str):
        super().__init__()
        model = whisper.load_model(whisper_model_name)
        self.encoder = model.encoder
        self.n_mels = model.dims.n_mels
        self.n_blocks = len(self.encoder.blocks)

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self._block_outputs: Dict[str, torch.Tensor] = {}
        for idx, block in enumerate(self.encoder.blocks):
            block.register_forward_hook(self._make_hook(f"encoder_block_{idx}"))

    def _make_hook(self, name: str):
        def hook_fn(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            self._block_outputs[name] = out
        return hook_fn

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        mel = whisper.log_mel_spectrogram(waveform, n_mels=self.n_mels).to(waveform.device)
        mel = whisper.pad_or_trim(mel, 3000)

        self._block_outputs.clear()
        encoder_out = self.encoder(mel)

        embeddings: Dict[str, torch.Tensor] = {}
        for name, feats in self._block_outputs.items():
            embeddings[name] = feats.flatten(start_dim=1)
        embeddings["ln_post"] = encoder_out.flatten(start_dim=1)

        self._block_outputs.clear()
        return embeddings


# ---------------------------------------------------------------------------
# Generic layerwise evaluation loop
# ---------------------------------------------------------------------------

def evaluate_layerwise(
    encoder: WhisperLayerwiseEncoder,
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
# Collate (shared across all three tasks)
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
        resample_sr=WHISPER_SR,
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
        target_sr=WHISPER_SR, sig_length=int(WHISPER_SR * 2),
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
        target_sr=WHISPER_SR,
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
    """Return list of (task, model_key) tuples — one per SLURM array element."""
    jobs = []
    for task in TASK_NAMES:
        for model_key in WHISPER_MODELS:
            jobs.append((task, model_key))
    return jobs


def job_csv_path(out_dir: Path, task: str, model_key: str) -> Path:
    return out_dir / f"zero_shot_{task}_whisper_{model_key}_layerwise.csv"


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------

def gather(out_dir: Path) -> Dict[str, pd.DataFrame]:
    """Concatenate per-job CSVs into one DataFrame per task.

    Returns a dict ``{task_name: DataFrame}`` and writes merged CSVs.
    """
    result: Dict[str, pd.DataFrame] = {}
    for task in TASK_NAMES:
        parts = []
        for model_key in WHISPER_MODELS:
            p = job_csv_path(out_dir, task, model_key)
            if p.exists():
                parts.append(pd.read_csv(p))
                print(f"  read {p}  ({len(parts[-1])} rows)")
            else:
                print(f"  MISSING {p}")
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged_path = out_dir / f"zero_shot_{task}_whisper_layerwise.csv"
            merged.to_csv(merged_path, index=False)
            print(f"  → {merged_path}  ({len(merged)} rows)")
            result[task] = merged
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layerwise Whisper zero-shot evaluation",
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

    jobs = build_job_list()

    # --list_jobs: print mapping and exit
    if args.list_jobs:
        for i, (task, model_key) in enumerate(jobs):
            print(f"  {i}: task={task}  model={model_key}")
        print(f"\nTotal jobs: {len(jobs)}  (use --array=0-{len(jobs) - 1})")
        return

    # --gather: merge CSVs and exit
    if args.gather:
        gather(out_dir)
        return

    # Resolve job_id
    job_id = args.job_id
    if job_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is not None:
            job_id = int(env_id)
        else:
            parser.error(
                "Provide --job_id or set SLURM_ARRAY_TASK_ID"
            )

    task, model_key = jobs[job_id]
    whisper_name = WHISPER_MODELS[model_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Job {job_id}: task={task}  model={whisper_name}")

    print(f"Loading Whisper {whisper_name} …")
    encoder = WhisperLayerwiseEncoder(whisper_name).to(device)
    print(f"  {encoder.n_blocks} encoder blocks")

    runner = TASK_RUNNERS[task]
    kwargs = dict(encoder=encoder, model_name=f"Whisper {whisper_name}", device=device)
    if args.n_examples is not None:
        kwargs["n_examples"] = args.n_examples

    records = runner(**kwargs)
    df = pd.DataFrame(records)

    csv_path = job_csv_path(out_dir, task, model_key)
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
