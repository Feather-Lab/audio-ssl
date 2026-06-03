# Release Manifest

This branch is prepared for sharing code associated with the anonymous CCN 2026
submission on contrastive-equivariant self-supervised learning for audio.

## Keep

- `README.md`, `LICENSE`, `setup.py`, `environment.yml`, `default_paths.py`,
  `figure_utils.py`, and this manifest.
- Core model/evaluation code:
  - `lightning_scripts/` — CE-SSL training (`train.py`), transfer/zero-shot eval,
    parameter decoding, and plotting entry points used by SLURM scripts.
  - `audio_ssl/`, `robustness/`, `analysis_scripts/`, `fmri_analysis/`
  - `data_scripts/extract_nsynth.py` — NSynth tarball helper referenced by eval loaders
  - `data_scripts/extract_demo_audio_batch.py` — optional JSIN/Audionoise clip extractor for the audio-transforms demo
  - `notebooks/demo_notebooks/demo_audio_batch/` — small pre-extracted speech/noise waveforms for `audio_transforms_demo.ipynb`
- `byol-a/` — BYOL-A baseline eval (`byol-a/config.yaml`); vendored without nested
  `.git` or `test/`.
- `model_checkpoints/` — not bundled; directory may exist locally with trained weights.
- `model_configs/` — 24 release YAMLs (CE-SSL λ-sweep, supervised baselines,
  `audiomae_pretrained_natsounds.yaml`, `whisper_pretrained_*.yaml`); no `old_configs/`,
  `barlow_search/`, `mmcr_search/`, or `equi_lmbda_search/`.
- `train_config_manifests/cochdnn9_sup_and_ssl_eval_configs.pkl` — NSynth linear-eval
  array manifest (relative paths).
- Release SLURM scripts in `slurm_scripts/` (21 scripts):
  - `extract_demo_audio_batch.sh`
  - `train_cochdnn_ce_ssl.sh`
  - `eval_cochdnn_layerwise_zero_shot.sh`
  - `eval_audiomae_layerwise_zero_shot.sh`
  - `eval_whisper_layerwise_zero_shot.sh`
  - `eval_esc_50.sh`
  - `eval_esc_50_byola.sh`
  - `eval_audiomae_esc50.sh`
  - `eval_whisper_esc50.sh`
  - `eval_speech_commands_transfer.sh`
  - `eval_audiomae_speech_commands.sh`
  - `eval_whisper_speech_commands.sh`
  - `eval_nsynth_linear_kell2018.sh`
  - `eval_nsynth_byola.sh`
  - `eval_audiomae_nsynth.sh`
  - `eval_whisper_nsynth.sh`
  - `run_param_decoding_kell2018.sh`
  - `run_param_decoding_audiomae.sh`
  - `run_param_decoding_whisper.sh`
  - `get_model_165_natsound_acts.sh`
  - `eval_audiomae_natsound_acts.sh`
- Demo notebooks: `notebooks/demo_notebooks/`
- Paper figure/analysis notebooks (under `notebooks/`):
  - `figure_2.ipynb`
  - `figure_2_scaled_plus_audiomae.ipynb`
  - `figure_3_parameter_decoding.ipynb`
  - `figure_3_parameter_decoding_w_audiomae.ipynb`
  - `figure_3_zero_shot.ipynb`
  - `figure_3_zero_shot_no_whisper_audiomae.ipynb`
  - `figure_3_zero_shot_with_whisper_and_audiomae.ipynb`
  - `figure_4_plot_fmri_components.ipynb`
  - `dev_layer_selection_analysis.ipynb`
  - `zero_shot_mandarin_tone_discrimination.ipynb`
  - `zero_shot_nsynth_melody_match.ipynb`
  - `zero_shot_speech_commands_level_discrimination.ipynb`

## Generated At Run Time (not in git)

Scripts write outputs under paths such as `outLogs/`, `results_dfs/`,
`eval_jsin_results/`, `eval_nsynth_results/`, `parameter_decoding_v2/`,
`parameter_decoding_audiomae/`, `fmri_analysis_model_features/`, and SLURM logs.
Re-run the corresponding `slurm_scripts/` jobs or Python entry points to recreate
them before opening figure notebooks.

## Removed From `to_share`

- Development/debug notebooks (`debug_*`, most `dev_*` except
  `dev_layer_selection_analysis`, `make_*`, exploratory plots, old zero-shot variants).
- Legacy JSIN/ImageNet eval scripts and shell wrappers (`eval_jsin*.py`,
  root-level `eval_*.sh`, `jupyter*.sh`, submodule `auditory_brain_dnn`).
- Hyperparameter-search config trees (`barlow_search/`, `mmcr_search/`,
  `equi_lmbda_search/`) and 18 pilot `train_config_manifests/*.pkl` files.
- `lightning_scripts/eval_jsin*.py`, `imagenet_dataset.py`,
  `lightning_ssl_imagenet.py`.
- Local caches and artifacts: `wandb/`, `lightning_logs/`, `esc_analysis/`,
  `dev_matched_supervised/`, `exp/`, ImageNet `assets/`, `talk_figs/`,
  `CCN_2026_figs/`, cached notebook CSVs under `notebooks/results_dfs/`.
- Duplicate configs (`* copy.yaml`), stale tests (`tests/`, `byol-a/test/`,
  root `test_cochdnn.py`).

## Data And Checkpoints

Full paper reproduction requires datasets and checkpoints that are not bundled in
this repository. Public datasets should be downloaded from their source projects;
restricted or lab-local assets should be supplied through environment variables
documented in `README.md` and `default_paths.py`.
