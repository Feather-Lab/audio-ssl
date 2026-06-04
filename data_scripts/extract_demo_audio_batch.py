#!/usr/bin/env python3
"""
Extract one MatchedSpeechInNoiseDatasetBatched training element for the audio
transforms demo notebook (idx=0, batch_size=1, blocked_batches=True).

Requires HDF5 paths (environment variables or CLI overrides):
  COCHDNN_JSIN_TRAIN_H5
  COCHDNN_AUDIONOISE_TRAIN_H5

Writes float32 waveforms at 20 kHz to notebooks/demo_notebooks/demo_audio_batch/:
  speech_1.npy, speech_2.npy — pre matched_random_crop (after length swap)
  noise_1.npy, noise_2.npy — full noise clips before RandomCrop(40000)
  metadata.json — indices, lengths, extraction seed
  *.wav — optional listening copies of the same clips
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "notebooks" / "demo_notebooks" / "demo_audio_batch"
TARGET_SR = 20_000
EXTRACTION_SEED = 2024
DATASET_IDX = 0
BATCH_SIZE = 1


def _resolve_h5_path(path, env_var):
    if path:
        resolved = os.path.expanduser(os.path.expandvars(str(path)))
    elif env_var in os.environ:
        resolved = os.path.expanduser(os.path.expandvars(os.environ[env_var]))
    else:
        return None
    if "$" in resolved:
        raise RuntimeError(f"Unresolved environment variable in HDF5 path: {resolved}")
    return Path(resolved)


def _as_float32_mono(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if wav.ndim != 1:
        raise ValueError(f"Expected 1-D waveform, got shape {wav.shape}")
    return wav


def _maybe_resample(wav: np.ndarray, in_sr: int, out_sr: int = TARGET_SR) -> np.ndarray:
    if in_sr == out_sr:
        return wav
    import torch
    from torchaudio.transforms import Resample

    t = torch.from_numpy(wav).unsqueeze(0)
    t = Resample(orig_freq=in_sr, new_freq=out_sr)(t)
    return t.squeeze(0).numpy().astype(np.float32)


def _infer_sample_rate(h5_path: Path, metadata: pd.DataFrame) -> int:
    for col in ("sample_rate", "sr", "fs", "sampling_rate"):
        if col in metadata.columns:
            val = metadata[col].dropna().iloc[0]
            return int(val)
    with h5py.File(h5_path, "r") as f:
        for key in ("sample_rate", "sr"):
            if key in f.attrs:
                return int(f.attrs[key])
    return TARGET_SR


def extract_batch_element(
    speech_h5: Path,
    noise_h5: Path,
    out_dir: Path,
    seed: int = EXTRACTION_SEED,
    dataset_idx: int = DATASET_IDX,
    write_wav: bool = True,
) -> dict:
    """Mirror MatchedSpeechInNoiseDatasetBatched.__getitem__ through pre-transform loads."""
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    with h5py.File(speech_h5, "r", swmr=True) as speech_files, h5py.File(
        noise_h5, "r", swmr=True
    ) as noise_files:
        speech_metadata = pd.read_hdf(speech_h5).dropna()
        noise_metadata = pd.read_hdf(noise_h5).dropna()
        num_noise_files = len(noise_metadata)

        speech_sr = _infer_sample_rate(speech_h5, speech_metadata)
        noise_sr = _infer_sample_rate(noise_h5, noise_metadata)

        start = dataset_idx * BATCH_SIZE * 2
        end = start + BATCH_SIZE * 2
        speech_ixs = np.arange(start, end)

        noise_idx = np.random.randint(num_noise_files - BATCH_SIZE * 2)
        noise_ixs = np.arange(noise_idx, noise_idx + BATCH_SIZE * 2)

        speech = speech_files["ndarray_data"]["signal"][speech_ixs]
        noise = noise_files["ndarray_data"]["signal"][noise_ixs]

        speech_batch_ixs = np.random.permutation(speech.shape[0])
        noise_batch_ixs = np.random.permutation(noise.shape[0])

        speech_1_ix, speech_2_ix = speech_batch_ixs[0], speech_batch_ixs[1]
        noise_1_ix, noise_2_ix = noise_batch_ixs[0], noise_batch_ixs[1]

        speech_label_1_ix = int(speech_ixs[speech_1_ix])
        speech_label_2_ix = int(speech_ixs[speech_2_ix])
        noise_label_1_ix = int(noise_ixs[noise_1_ix])
        noise_label_2_ix = int(noise_ixs[noise_2_ix])

        speech_1 = _as_float32_mono(speech[speech_1_ix])
        speech_2 = _as_float32_mono(speech[speech_2_ix])
        length_swapped = False
        if len(speech_1) > len(speech_2):
            speech_1, speech_2 = speech_2, speech_1
            speech_label_1_ix, speech_label_2_ix = speech_label_2_ix, speech_label_1_ix
            length_swapped = True

        speech_1 = _maybe_resample(speech_1, speech_sr)
        speech_2 = _maybe_resample(speech_2, speech_sr)
        noise_1 = _maybe_resample(_as_float32_mono(noise[noise_1_ix]), noise_sr)
        noise_2 = _maybe_resample(_as_float32_mono(noise[noise_2_ix]), noise_sr)

    arrays = {
        "speech_1": speech_1,
        "speech_2": speech_2,
        "noise_1": noise_1,
        "noise_2": noise_2,
    }
    for name, wav in arrays.items():
        np.save(out_dir / f"{name}.npy", wav)

    metadata = {
        "description": "One MatchedSpeechInNoiseDatasetBatched element (batch_size=1, blocked_batches=True)",
        "dataset_idx": dataset_idx,
        "batch_size": BATCH_SIZE,
        "blocked_batches": True,
        "extraction_seed": seed,
        "sample_rate_hz": TARGET_SR,
        "speech_h5_inferred_sr": speech_sr,
        "noise_h5_inferred_sr": noise_sr,
        "speech_h5_path": str(speech_h5),
        "noise_h5_path": str(noise_h5),
        "speech_row_indices": [speech_label_1_ix, speech_label_2_ix],
        "noise_row_indices": [noise_label_1_ix, noise_label_2_ix],
        "speech_block_range": [int(start), int(end)],
        "noise_block_start_index": int(noise_idx),
        "length_swapped": length_swapped,
        "lengths_samples": {k: int(len(v)) for k, v in arrays.items()},
        "note": "Speech/noise saved before matched_random_crop and RandomCrop; notebook applies CE-SSL transforms.",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if write_wav:
        import soundfile as sf

        for name, wav in arrays.items():
            sf.write(out_dir / f"{name}.wav", wav, TARGET_SR)

    return metadata


def _print_setup_help():
    print(
        "JSIN / Audionoise train HDF5 paths are not configured.\n\n"
        "Set environment variables (see README.md):\n"
        "  export COCHDNN_JSIN_TRAIN_H5=/path/to/train_speech.h5\n"
        "  export COCHDNN_AUDIONOISE_TRAIN_H5=/path/to/train_noise.h5\n\n"
        "Then submit from the repository root:\n"
        "  sbatch slurm_scripts/extract_demo_audio_batch.sh\n\n"
        "Or pass paths explicitly (cluster job only; do not run locally):\n"
        "  python data_scripts/extract_demo_audio_batch.py \\\n"
        "    --speech_h5 /path/to/train_speech.h5 \\\n"
        "    --noise_h5 /path/to/train_noise.h5"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech_h5", type=str, default=None)
    parser.add_argument("--noise_h5", type=str, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=EXTRACTION_SEED)
    parser.add_argument("--dataset_idx", type=int, default=DATASET_IDX)
    parser.add_argument("--no_wav", action="store_true", help="Skip writing .wav files")
    args = parser.parse_args()

    speech_h5 = _resolve_h5_path(args.speech_h5, "COCHDNN_JSIN_TRAIN_H5")
    noise_h5 = _resolve_h5_path(args.noise_h5, "COCHDNN_AUDIONOISE_TRAIN_H5")

    if speech_h5 is None or noise_h5 is None:
        _print_setup_help()
        return 1

    for label, path in (("speech", speech_h5), ("noise", noise_h5)):
        if not path.is_file():
            print(f"Error: {label} HDF5 not found: {path}")
            _print_setup_help()
            return 1

    meta = extract_batch_element(
        speech_h5=speech_h5,
        noise_h5=noise_h5,
        out_dir=args.out_dir,
        seed=args.seed,
        dataset_idx=args.dataset_idx,
        write_wav=not args.no_wav,
    )
    print(f"Wrote demo batch to {args.out_dir}")
    print(json.dumps(meta["lengths_samples"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
