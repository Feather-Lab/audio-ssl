# Demo audio batch for `audio_transforms_demo.ipynb`

Small pre-extracted speech/noise waveforms (one `MatchedSpeechInNoiseDatasetBatched` element).

Regenerate on the cluster from the repository root when train HDF5 paths are available:

```bash
export COCHDNN_JSIN_TRAIN_H5=/path/to/train_speech.h5
export COCHDNN_AUDIONOISE_TRAIN_H5=/path/to/train_noise.h5
sbatch slurm_scripts/extract_demo_audio_batch.sh
```

Do not run `data_scripts/extract_demo_audio_batch.py` locally — submit the SLURM job above.
This writes `speech_*.npy`, `noise_*.npy`, `metadata.json`, and optional `.wav` files here.
