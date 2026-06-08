"""
Shared utilities for zero-shot evaluation notebooks:
- zero_shot_speech_commands_level_discrimination
- zero_shot_nsynth_melody_match
- zero_shot_mandarin_tone_discrimination
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm.auto import tqdm

from default_paths import WORKING_DIRECTORY
from lightning_scripts.byola_lightning_module import BYOLAModule
from lightning_scripts.utils.model_build_utils import get_model, get_checkpoint_path

# Default sample rate for CochDNN / robustness models
MODEL_SR = 20_000
BYOLA_SR = 16_000
DEFAULT_SIG_LENGTH = 40_000

config_dir = WORKING_DIRECTORY

# ---------------------------------------------------------------------------
# CochCNN9 model registry for single-model loading (layerwise evaluation)
# ---------------------------------------------------------------------------

COCHDNN_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ssl_0.0": {
        "config": "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_0e-01.yaml",
        "supervised": False,
        "display_name": "CochCNN9 ssl λ=0.0",
    },
    "ssl_0.1": {
        "config": "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_1e-01.yaml",
        "supervised": False,
        "display_name": "CochCNN9 ssl λ=0.1",
    },
    "ssl_0.2": {
        "config": "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_2e-01.yaml",
        "supervised": False,
        "display_name": "CochCNN9 ssl λ=0.2",
    },
    "ssl_0.3": {
        "config": "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_3e-01.yaml",
        "supervised": False,
        "display_name": "CochCNN9 ssl λ=0.3",
    },
    "ssl_0.4": {
        "config": "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_4e-01.yaml",
        "supervised": False,
        "display_name": "CochCNN9 ssl λ=0.4",
    },
    "ssl_0.5": {
        "config": "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_5e-01.yaml",
        "supervised": False,
        "display_name": "CochCNN9 ssl λ=0.5",
    },
    "scaled_ssl_0.0": {
        "config": "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_0e-01_audioset_only.yaml",
        "supervised": False,
        "display_name": "CochCNN9 scaled ssl λ=0.0",
    },
    "scaled_ssl_0.5": {
        "config": "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_5e-01_audioset_only.yaml",
        "supervised": False,
        "display_name": "CochCNN9 scaled ssl λ=0.5",
    },
    "word": {
        "config": "model_configs/supervised_models/word_kell2018_MatchedDataset_LARS.yaml",
        "supervised": True,
        "display_name": "CochCNN9 supervised word",
    },
    "audioset": {
        "config": "model_configs/supervised_models/audioset_kell2018_MatchedDataset_LARS.yaml",
        "supervised": True,
        "display_name": "CochCNN9 supervised audioset",
    },
    "multitask": {
        "config": "model_configs/supervised_models/kell2018_word_speaker_audioset_MatchedDataset_LARS.yaml",
        "supervised": True,
        "display_name": "CochCNN9 supervised multi-task",
    },
    "scaled_supervised": {
        "config": "model_configs/supervised_models/kell2018_audioset_unbalanced_supervised.yaml",
        "supervised": True,
        "display_name": "CochCNN9 scaled supervised",
    },
}


def load_single_cochdnn_model(
    model_key: str,
    device: str = "cuda",
) -> tuple:
    """Load a single CochCNN9 model and return a layerwise encoder callable.

    Uses the existing ``ModelWithFrontEnd`` wrapper which already supports
    ``with_latent=True`` for both SSL and supervised models.

    Args:
        model_key: Key into :data:`COCHDNN_MODEL_REGISTRY`.
        device: Torch device string.

    Returns:
        encoder: callable ``(waveform: Tensor) -> Dict[str, Tensor]``
            accepting ``(B, 1, T)`` waveforms at 20 kHz and returning
            flattened ``(B, D)`` activations keyed by layer name.
        display_name: Human-readable model name for CSV output.
        layer_names: Ordered list of available layer names.
    """
    from lightning_scripts.lightning_classifier_matched_speech_in_noise import (
        LitWordAudioSetModel,
    )
    from lightning_scripts.lightning_ssl_matched_speech_in_noise import LitAudioSSL

    info = COCHDNN_MODEL_REGISTRY[model_key]
    config_path = config_dir / info["config"]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    checkpoint_path = get_checkpoint_path(config_path)

    if info["supervised"]:
        lit_module = LitWordAudioSetModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, config=config, strict=True,
        ).eval()
    else:
        lit_module = LitAudioSSL.load_from_checkpoint(
            checkpoint_path=checkpoint_path, config=config, strict=True,
        ).eval()

    layer_names = list(lit_module.metamer_layers)
    model_with_frontend = lit_module.model.eval()

    for p in model_with_frontend.parameters():
        p.requires_grad = False

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model_with_frontend = model_with_frontend.to(dev)

    @torch.no_grad()
    def encoder(waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, _, all_outputs = model_with_frontend(
            waveform, with_latent=True, fake_relu=False,
        )
        return {
            k: v.flatten(start_dim=1)
            for k, v in all_outputs.items()
            if isinstance(v, torch.Tensor)
        }

    return encoder, info["display_name"], layer_names

def get_cochdnn9_models(
    layer_out: str = "relufc",
    device: str = "cuda",
) -> tuple[Dict[str, torch.nn.Module], Dict[str, str]]:
    """
    Load CochCNN9 and BYOL-A models for zero-shot tasks.

    Returns:
        models: dict mapping internal key -> model
        model_name_map: dict mapping internal key -> display name
    """
    ssl_configs = [
        ("ssl_λ=0.0", "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_0e-01.yaml"),
        ("ssl_λ=0.1", "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_1e-01.yaml"),
        ("ssl_λ=0.2", "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_2e-01.yaml"),
        ("ssl_λ=0.3", "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_3e-01.yaml"),
        ("ssl_λ=0.4", "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_4e-01.yaml"),
        ("ssl_λ=0.5", "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_5e-01.yaml"),
    ]
    scaled_ssl_configs = [
        ("scaled_ssl_λ=0.0", "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_0e-01_audioset_only.yaml"),
        ("scaled_ssl_λ=0.5", "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_5e-01_audioset_only.yaml"),
    ]
    supervised_configs = [
        ("word", "model_configs/supervised_models/word_kell2018_MatchedDataset_LARS.yaml"),
        ("audioset", "model_configs/supervised_models/audioset_kell2018_MatchedDataset_LARS.yaml"),
        ("multitask", "model_configs/supervised_models/kell2018_word_speaker_audioset_MatchedDataset_LARS.yaml"),
        ("unbal_audioset", "model_configs/supervised_models/kell2018_audioset_unbalanced_supervised.yaml"),
    ]

    models: Dict[str, torch.nn.Module] = {}
    model_name_map: Dict[str, str] = {}

    for key, rel_path in ssl_configs:
        models[key] = get_model(config_dir / rel_path, layer_out=layer_out)
        model_name_map[key] = f"CochCNN9 ssl λ={key.split('=')[1]}"

    for key, rel_path in scaled_ssl_configs:
        models[key] = get_model(config_dir / rel_path, layer_out=layer_out)
        model_name_map[key] = f"CochCNN9 scaled ssl λ={key.split('=')[1]}"

    for key, rel_path in supervised_configs:
        models[key] = get_model(config_dir / rel_path, supervised=True, layer_out=layer_out)

    # add random init model 
    models["random_init"] = get_model(config_dir / "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_0e-01.yaml", random=True)
    
    model_name_map["random_init"] = "CochCNN9 random weights" 
    model_name_map["word"] = "CochCNN9 supervised word"
    model_name_map["audioset"] = "CochCNN9 supervised audioset"
    model_name_map["multitask"] = "CochCNN9 supervised multi-task"
    model_name_map["unbal_audioset"] = "CochCNN9 scaled supervised"

    with open(config_dir / "byol-a/config.yaml", "r") as f:
        byola_config = yaml.load(f, Loader=yaml.FullLoader)
    # use last conv for equitable comparison 
    byola_config['classifier_layer'] = "features.10"
    models["byol-a"] = BYOLAModule(byola_config).cuda().eval()
    model_name_map["byol-a"] = "byol-a"

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    for model in models.values():
        model.to(dev)

    return models, model_name_map


def encode_audio(model: torch.nn.Module, audio_batch: torch.Tensor) -> torch.Tensor:
    """Encode a batch of audio to embeddings. Handles (B,C,T) and flattens to (B,D)."""
    model.eval()
    with torch.no_grad():
        z = model(audio_batch)
        if z.ndim > 2:
            z = z.flatten(start_dim=1)
    return z


def _prepare_triplet_waves(
    clips: Dict[str, torch.Tensor],
    sr: int,
    device: torch.device,
    model: torch.nn.Module,
    model_sr: int = MODEL_SR,
) -> Dict[str, torch.Tensor]:
    """Resample clips to model sample rate and move to device."""
    waves: Dict[str, torch.Tensor] = {}
    for k, v in clips.items():
        w = v
        if isinstance(model, BYOLAModule):
            if sr != BYOLA_SR:
                w = torchaudio.functional.resample(w.unsqueeze(0), sr, BYOLA_SR).squeeze(0)
        elif sr != model_sr:
            w = torchaudio.functional.resample(w.unsqueeze(0), sr, model_sr).squeeze(0)
        waves[k] = w.to(device)
    return waves


def batched_pearson_corrcoef(x, y, dim=-1):
    """
    Compute Pearson correlation along a dimension.
    
    Args:
        x: tensor of shape (N, ...)
        y: tensor of shape (N, ...)
        dim: dimension along which to compute correlation
    
    Returns:
        Correlation coefficient(s)
    """
    mean_x = torch.mean(x, dim=dim, keepdim=True)
    mean_y = torch.mean(y, dim=dim, keepdim=True)
    
    xm = x - mean_x
    ym = y - mean_y
    
    r_num = torch.sum(xm * ym, dim=dim)
    r_den = torch.sqrt(torch.sum(xm ** 2, dim=dim) * torch.sum(ym ** 2, dim=dim))
    
    r = r_num / r_den
    return r



def embedding_triplet_metrics(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
) -> Dict[str, float]:
    """Compute distance metrics on pre-computed (anchor, positive, negative) embeddings.

    Each input should be shaped (B, D).  Returns L2, squared-L2, cosine,
    and Pearson-correlation distances plus binary judgements.
    """
    pos_l2 = torch.norm(z_anchor - z_pos, p=2).item()
    neg_l2 = torch.norm(z_anchor - z_neg, p=2).item()
    pos_l2_sq = pos_l2**2
    neg_l2_sq = neg_l2**2

    pos_cos = F.cosine_similarity(z_anchor, z_pos).item()
    neg_cos = F.cosine_similarity(z_anchor, z_neg).item()

    pos_r = batched_pearson_corrcoef(z_anchor, z_pos, dim=-1).item()
    neg_r = batched_pearson_corrcoef(z_anchor, z_neg, dim=-1).item()

    return {
        "pos_l2": pos_l2,
        "neg_l2": neg_l2,
        "pos_l2_sq": pos_l2_sq,
        "neg_l2_sq": neg_l2_sq,
        "l2_judgement": int(pos_l2 < neg_l2),
        "sqr_l2_judgement": int(pos_l2_sq < neg_l2_sq),
        "judgement_pos_lt_neg": int(pos_l2 < neg_l2),
        "pos_cos": pos_cos,
        "neg_cos": neg_cos,
        "cos_judgement": int(pos_cos > neg_cos),
        "pos_r": pos_r,
        "neg_r": neg_r,
        "r_judgement": int(pos_r > neg_r),
    }


def distance_metrics_on_triplet(
    model: torch.nn.Module,
    clips: Dict[str, torch.Tensor],
    sr: int,
    device: Optional[torch.device] = None,
    model_sr: int = MODEL_SR,
    include_cosine: bool = False,
) -> Dict[str, float]:
    """
    Compute embedding-space distance metrics for (anchor, positive, negative).

    Encodes clips with *model*, then delegates to ``embedding_triplet_metrics``.
    """
    if device is None:
        device = next(model.parameters()).device
    waves = _prepare_triplet_waves(clips, sr, device, model, model_sr)

    if isinstance(model, BYOLAModule):
        anchor_batch = waves["anchor"].unsqueeze(0)
        positive_batch = waves["positive"].unsqueeze(0)
        negative_batch = waves["negative"].unsqueeze(0)
    else:
        anchor_batch = waves["anchor"].unsqueeze(0).unsqueeze(1)
        positive_batch = waves["positive"].unsqueeze(0).unsqueeze(1)
        negative_batch = waves["negative"].unsqueeze(0).unsqueeze(1)

    z_anchor = encode_audio(model, anchor_batch)
    z_pos = encode_audio(model, positive_batch)
    z_neg = encode_audio(model, negative_batch)

    return embedding_triplet_metrics(z_anchor, z_pos, z_neg)


def run_triplet_evaluation(
    models: Dict[str, torch.nn.Module],
    model_name_map: Dict[str, str],
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    model_sr: int = MODEL_SR,
    include_cosine: bool = False,
    normalize_name_fn: Optional[Callable[[str], str]] = None,
    extract_triplet_meta_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    desc: str = "Evaluating triplets",
) -> pd.DataFrame:
    """
    Run all models on a triplet dataloader and return a single DataFrame.

    Each batch from dataloader should be (clips, sr, triplet_meta) where
    clips is dict with 'anchor', 'positive', 'negative' tensors.

    If normalize_name_fn is provided, it is applied to the display name
    (e.g. figure_utils.normalize_model_name). If extract_triplet_meta_fn is
    provided, it receives triplet_meta and returns a dict of columns to add
    to each row (e.g. speaker, tone, batch_idx, interval, etc.).
    """
    if device is None:
        device = next(next(iter(models.values())).parameters()).device
    all_results: List[Dict[str, Any]] = []

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=desc):
        clips, sr, triplet_meta = batch
        sr_val = int(sr)

        for name, model in models.items():
            metrics = distance_metrics_on_triplet(
                model, clips, sr_val, device=device, model_sr=model_sr, include_cosine=include_cosine
            )
            display_name = model_name_map.get(name, name)
            if normalize_name_fn is not None:
                display_name = normalize_name_fn(display_name)
            row: Dict[str, Any] = {
                "model": name,
                "model_name": display_name,
                "batch_idx": batch_idx,
                **metrics,
            }
            if extract_triplet_meta_fn is not None and triplet_meta is not None:
                row.update(extract_triplet_meta_fn(triplet_meta))
            all_results.append(row)

    return pd.DataFrame(all_results)


class CenterCropOrPad:
    """Center-crop or center-pad a 1D signal to a target length."""

    def __init__(self, sig_length: int):
        self.sig_length = sig_length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < self.sig_length:
            pad_dur = (self.sig_length - x.shape[0]) // 2 + 1
            x = F.pad(x, (pad_dur, pad_dur), "constant", 0)
        start_idx = int((x.shape[0] - self.sig_length) / 2)
        x = x[start_idx : start_idx + self.sig_length]
        return x


# Optional robustness-based helpers (used by speech-commands level task)
try:
    import robustness.audio_functions.audio_transforms as at

    def get_dbSPL_normalizer(db_spl: int = 60):
        return at.DBSPLNormalizeForegroundAndBackground(db_spl)

    DEFAULT_DB_SPL = 60
    default_crop_or_pad = CenterCropOrPad(DEFAULT_SIG_LENGTH)
    default_set_dbSPL = get_dbSPL_normalizer(DEFAULT_DB_SPL)
except ImportError:
    at = None
    get_dbSPL_normalizer = None
    DEFAULT_DB_SPL = 60
    default_crop_or_pad = CenterCropOrPad(DEFAULT_SIG_LENGTH)
    default_set_dbSPL = None
