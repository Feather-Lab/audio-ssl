# CE-SSL Audio Representations

Code for an anonymous CCN 2026 submission on contrastive-equivariant
self-supervised learning (CE-SSL) for general-purpose audio representations.
The project adapts a contrastive-equivariant objective to audio and evaluates
representations with transfer probes, zero-shot triplet judgments, parameter
decoding, and fMRI prediction.

## What Is Included

- CE-SSL/iSSL training and evaluation code for CochCNN9-style audio encoders.
- Linear-probe evaluation scripts for ESC-50, Speech Commands, NSynth, and
  restricted WSN/JSIN word tasks.
- Zero-shot triplet evaluation for intensity, melody, and Mandarin tone tasks.
- Parameter-decoding and fMRI-analysis entry points used for paper analyses.
- Minimal demo notebooks in `notebooks/demo_notebooks/`.

See `RELEASE_MANIFEST.md` for the intended release surface.

Release development lives on branch `to_share` (not merged to `main` until review).

## Repository Layout

- `lightning_scripts/` — CE-SSL training (`train.py`), transfer and zero-shot eval, parameter decoding, `optimizers.py` (LARS / cosine warmup)
- `robustness/` — vendored lab fork (matched audio transforms, Kell2018/ResNet models)
- `slurm_scripts/` — cluster entry points for training, evaluation, and demo asset extraction
- `model_configs/` — release YAML configs; `train_config_manifests/` — NSynth eval manifest
- `notebooks/demo_notebooks/` — minimal demos; `notebooks/*.ipynb` — figure analyses (see manifest)
- `fmri_analysis/`, `analysis_scripts/`, `parameter_decoding/`, `default_paths.py`
- Submodules: `auditory_brain_dnn/`, `byol-a/`

## Setup

Clone with submodules. External dependencies live in git submodules; the
modified `robustness/` tree is vendored in the parent repository (required for CE-SSL
audio transforms and encoder code — not a submodule):

```bash
git clone --recurse-submodules <repo-url>
cd cochdnn
# or, after a plain clone:
git submodule update --init --recursive
```

Git submodules: `auditory_brain_dnn`, `byol-a` ([nttcslab/byol-a](https://github.com/nttcslab/byol-a)).
`robustness/` ships with the parent repo (lab-modified fork; see `RELEASE_MANIFEST.md`).
After init, place BYOL-A pretrained checkpoints under
`byol-a/pretrained_weights/` if needed (see upstream BYOL-A docs; weights are not
tracked in the parent repo).

Create the conda environment used by the retained scripts:

```bash
mamba env create -f environment.yml
mamba activate cochdnn_ssl_pl
pip install -e .
```

The release scripts activate `cochdnn_ssl_pl`. If you use a different
environment name, update the scripts locally or activate your environment before
running the Python commands directly.

## Cluster Execution

Compute-heavy jobs (training, evaluation sweeps, demo asset extraction, fMRI
feature extraction) must be submitted to the cluster with SLURM — do not run them
locally. From the repository root:

```bash
sbatch slurm_scripts/<script>.sh
```

Logs go to `outLogs/`. Set dataset and checkpoint paths via the environment
variables below before submitting; this release does not assume local GPUs or
full datasets on a laptop or interactive node.

## Quickstart

Start with the demo notebooks:

- `notebooks/demo_notebooks/audio_transforms_demo.ipynb` — CE-SSL matched transforms on a fixed real speech/noise batch (`demo_audio_batch/`)
- `notebooks/demo_notebooks/zero_shot_eval_demo.ipynb` — NSynth melody-match
  triplets with CE-SSL CochCNN9 (`COCHDNN_CHECKPOINT_DIR`, `COCHDNN_NSYNTH_DIR`)

The zero-shot demo writes a small CSV next to the notebook. The transform demo
loads bundled clips under `notebooks/demo_notebooks/demo_audio_batch/`; regenerate
that folder on the cluster with
`sbatch slurm_scripts/extract_demo_audio_batch.sh` when
`COCHDNN_JSIN_TRAIN_H5` and `COCHDNN_AUDIONOISE_TRAIN_H5` are set.

## Data And Checkpoints

Datasets and checkpoints are not bundled. Configure paths with environment
variables:

```bash
export COCHDNN_CHECKPOINT_DIR=/path/to/model_checkpoints
export COCHDNN_ESC50_DIR=/path/to/ESC-50-master
export COCHDNN_NSYNTH_DIR=/path/to/nsynth
export COCHDNN_TONE_PERFECT_DIR=/path/to/tone_perfect
export COCHDNN_JSIN_DIR=/path/to/JSIN_all_v3/subsets
export COCHDNN_JSIN_TRAIN_H5=/path/to/train_speech.h5
export COCHDNN_JSIN_VALID_H5=/path/to/valid_speech.h5
export COCHDNN_AUDIONOISE_TRAIN_H5=/path/to/train_noise.h5
export COCHDNN_AUDIONOISE_VALID_H5=/path/to/valid_noise.h5
```

`COCHDNN_JSIN_*` and fMRI assets refer to restricted datasets and are needed
only for the corresponding training, WSN/JSIN transfer, parameter-decoding, and
neural-prediction analyses.

## Example Commands

Submit from the repository root (see `slurm_scripts/` for the full set):

```bash
# CE-SSL training (one representative config; extend time/GPUs as needed)
sbatch slurm_scripts/train_cochdnn_ce_ssl.sh

# Zero-shot layerwise evaluation (array jobs; per-job gather is automatic)
sbatch slurm_scripts/eval_cochdnn_layerwise_zero_shot.sh
sbatch slurm_scripts/eval_audiomae_layerwise_zero_shot.sh
sbatch slurm_scripts/eval_whisper_layerwise_zero_shot.sh

# Transfer probes and parameter decoding
sbatch slurm_scripts/eval_esc_50.sh
sbatch slurm_scripts/eval_speech_commands_transfer.sh
sbatch slurm_scripts/eval_nsynth_linear_kell2018.sh
sbatch slurm_scripts/run_param_decoding_kell2018.sh

# Demo audio batch for audio_transforms_demo.ipynb
sbatch slurm_scripts/extract_demo_audio_batch.sh
```

For AudioMAE or Whisper baselines, use the corresponding `eval_audiomae_*`,
`eval_whisper_*`, and `run_param_decoding_*` scripts.

## Notes

The demos are intended to check mechanics and provide minimal reproducible
examples. Full paper reproduction requires the private/restricted datasets,
model checkpoints, and compute resources used for the reported experiments.
