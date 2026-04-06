"""
Shared utilities for loading and running the pretrained AudioMAE encoder
(hance-ai/audiomae) with per-layer activation extraction.

AudioMAE is a ViT-Base masked autoencoder pretrained on AudioSet.
Architecture: 16x16 patch embedding -> 12 transformer blocks -> layer norm.
Input: 128-bin Kaldi fbank mel spectrogram, padded/trimmed to 1024 frames (10s max).
Patches: 64 time x 8 freq = 512 patches + 1 CLS token, each dim 768.

Reference:
  Huang et al., "Masked autoencoders that listen", NeurIPS 2022.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torchaudio
from torchaudio.compliance import kaldi

AUDIOMAE_SR = 16_000
AUDIOMAE_MAX_SECS = 10
AUDIOMAE_N_BLOCKS = 12
AUDIOMAE_DIM = 768
AUDIOMAE_FREQ_PATCHES = 8
AUDIOMAE_TIME_PATCHES = 64
AUDIOMAE_N_MELS = 128
AUDIOMAE_EXPECTED_FRAMES = 1024

AUDIOMAE_MEAN = -4.2677393
AUDIOMAE_STD = 4.5689974


def audiomae_layer_names_list(n_blocks: int = AUDIOMAE_N_BLOCKS) -> List[str]:
    """Canonical ordered layer names: ``block_0`` … ``block_{n-1}``, then ``norm``."""
    return [f"block_{i}" for i in range(n_blocks)] + ["norm"]


def validate_audiomae_layer(layer: str, n_blocks: int = AUDIOMAE_N_BLOCKS) -> str:
    """Return ``layer`` if it is a valid AudioMAE hook / norm name; else raise."""
    names = audiomae_layer_names_list(n_blocks)
    if layer not in names:
        raise ValueError(
            f"Invalid AudioMAE layer {layer!r}. Expected one of: {', '.join(names)}"
        )
    return layer


def audiomae_layer_from_slurm_index(
    task_id: int,
    n_blocks: int = AUDIOMAE_N_BLOCKS,
) -> str:
    """Map SLURM array task id to layer name (same order as ``eval_audiomae_*.sh``)."""
    names = audiomae_layer_names_list(n_blocks)
    if task_id < 0 or task_id >= len(names):
        raise ValueError(
            f"job_id / array index {task_id} out of range [0, {len(names) - 1}] "
            f"for AudioMAE ({len(names)} layers)."
        )
    return names[task_id]


def parse_audiomae_layer_str(
    layer_str: str,
    *,
    valid_layers: Optional[Sequence[str]] = None,
    n_blocks: int = AUDIOMAE_N_BLOCKS,
) -> str:
    """Normalize and validate a user-provided layer string (e.g. from CLI)."""
    s = layer_str.strip()
    if valid_layers is not None:
        if s not in valid_layers:
            raise ValueError(
                f"Invalid AudioMAE layer {s!r}. Expected one of: {list(valid_layers)}"
            )
        return s
    return validate_audiomae_layer(s, n_blocks=n_blocks)


def preprocess_waveform(
    waveform: torch.Tensor,
    sr: int = AUDIOMAE_SR,
) -> torch.Tensor:
    """Convert a waveform tensor to the mel spectrogram expected by AudioMAE.

    Args:
        waveform: (B, T) or (T,) raw audio at ``sr`` Hz.
        sr: sample rate of the input waveform (default 16 kHz).

    Returns:
        Mel spectrogram of shape (B, 1, 1024, 128).
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if sr != AUDIOMAE_SR:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=AUDIOMAE_SR)

    batch_mels = []
    for i in range(waveform.shape[0]):
        mel = kaldi.fbank(
            waveform[i : i + 1],
            num_mel_bins=AUDIOMAE_N_MELS,
            frame_length=25.0,
            frame_shift=10.0,
            htk_compat=True,
            use_energy=False,
            sample_frequency=AUDIOMAE_SR,
            window_type="hanning",
            dither=0.0,
        )  # (T_frames, 128)

        if mel.shape[0] > AUDIOMAE_EXPECTED_FRAMES:
            mel = mel[:AUDIOMAE_EXPECTED_FRAMES, :]
        elif mel.shape[0] < AUDIOMAE_EXPECTED_FRAMES:
            pad = AUDIOMAE_EXPECTED_FRAMES - mel.shape[0]
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))

        mel = (mel - AUDIOMAE_MEAN) / (AUDIOMAE_STD * 2)
        batch_mels.append(mel)

    mels = torch.stack(batch_mels, dim=0)  # (B, 1024, 128)
    return mels.unsqueeze(1)  # (B, 1, 1024, 128)


class AudioMAELayerwiseEncoder(nn.Module):
    """Wrap the pretrained AudioMAE encoder with hooks on every ViT block.

    Accepts raw 16 kHz waveform ``(B, T)`` or ``(B, 1, T)`` and returns
    ``Dict[layer_name, Tensor]`` with activations for every transformer
    block and the final layer-norm output.
    """

    def __init__(self, time_pool: bool = False):
        super().__init__()
        from transformers import AutoModel
        from transformers.modeling_utils import PreTrainedModel

        _orig_mark = PreTrainedModel.mark_tied_weights_as_initialized
        def _safe_mark(self, loading_info):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return _orig_mark(self, loading_info)
        PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark

        wrapper = AutoModel.from_pretrained(
            "hance-ai/audiomae", trust_remote_code=True
        )
        self.encoder = wrapper.encoder  # AudioMAEEncoder (VisionTransformer)

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.n_blocks = len(self.encoder.blocks)
        self.time_pool = time_pool

        self._block_outputs: Dict[str, torch.Tensor] = {}
        for idx, block in enumerate(self.encoder.blocks):
            block.register_forward_hook(self._make_hook(f"block_{idx}"))

    def _make_hook(self, name: str):
        def hook_fn(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            self._block_outputs[name] = out
        return hook_fn

    def _reshape_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N_patches, D) -> (B, D, freq, time) after removing CLS."""
        tokens = tokens[:, 1:, :]  # drop CLS
        B, _N, D = tokens.shape
        return tokens.reshape(B, AUDIOMAE_FREQ_PATCHES, AUDIOMAE_TIME_PATCHES, D).permute(
            0, 3, 1, 2
        )  # (B, D, freq, time)

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        sr: int = AUDIOMAE_SR,
    ) -> Dict[str, torch.Tensor]:
        """Run the encoder and return per-layer embeddings.

        Args:
            waveform: (B, T) or (B, 1, T) raw audio.
            sr: sample rate (default 16 kHz).

        Returns:
            Dict mapping layer name to tensor.  Shape depends on
            ``self.time_pool``:
              - time_pool=False: (B, D*freq*time) = (B, 393216)
              - time_pool=True:  (B, D*freq) = (B, 6144)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        mel = preprocess_waveform(waveform, sr=sr).to(waveform.device)

        self._block_outputs.clear()
        full_out = self.encoder.forward_features(mel)  # (B, 1+N, D)

        embeddings: Dict[str, torch.Tensor] = {}
        for name, tokens in self._block_outputs.items():
            feats = self._reshape_tokens(tokens)  # (B, D, freq, time)
            if self.time_pool:
                feats = feats.mean(dim=-1)  # (B, D, freq)
            embeddings[name] = feats.flatten(start_dim=1)

        norm_tokens = full_out  # after blocks + norm
        norm_feats = self._reshape_tokens(norm_tokens)
        if self.time_pool:
            norm_feats = norm_feats.mean(dim=-1)
        embeddings["norm"] = norm_feats.flatten(start_dim=1)

        self._block_outputs.clear()
        return embeddings

    @property
    def layer_names(self):
        return [f"block_{i}" for i in range(self.n_blocks)] + ["norm"]
