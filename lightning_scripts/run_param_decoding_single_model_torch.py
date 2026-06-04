"""Parameter decoding evaluation script for single model evaluation.

This script evaluates a single model's ability to decode augmentation parameters
from its representations. It supports both SSL and supervised models.
"""

import logging
import os
import pickle
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import robustness.audio_functions.audio_transforms as at
import torch
import yaml
from lightning import seed_everything
from tqdm import tqdm

from default_paths import JSIN_PATH, require_path

from lightning_scripts.jsinV3DataLoader_precombined_batched import (
    jsinV3_precombined_all_signals,
)
from lightning_scripts.lightning_classifier_matched_speech_in_noise import (
    LitWordAudioSetModel,
)
from lightning_scripts.lightning_ssl_matched_speech_in_noise import LitAudioSSL
from lightning_scripts.audiomae_encoder_utils import (
    AUDIOMAE_SR,
    AudioMAELayerwiseEncoder,
    parse_audiomae_layer_str,
)
from lightning_scripts.whisper_encoder_arch import (
    WhisperLayerwiseEncoder,
    parse_whisper_layer_str,
)

# Configure PyTorch
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Constants
JSIN_PATH = str(require_path(JSIN_PATH, "COCHDNN_JSIN_DIR", "JSIN/WSN dataset"))
CROP_LENGTH = 40_000  # 2 seconds at 20kHz
SAMPLE_RATE = 20_000
DBSPL_LEVEL = 60

# Augmentation parameter ranges
SNR_RANGE = [-10, 10]
PITCH_RANGE = [-0.5, 0.5]
TEMPO_RANGE = [-0.8, 1.2]
TIME_SHIFT_RANGE = [-0.250, 0.250]
FILTER_ORDER_RANGE = [1, 4]  # Discrete: 1, 2, 3, 4
BANDPASS_FREQ_LOW_RANGE = [4e1, 4e2]
BANDPASS_FREQ_HIGH_RANGE = [4e3, 10e3]


def _ensure_2d_tensor(values: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has shape [N, P]."""
    return values.unsqueeze(1) if values.ndim == 1 else values


def torch_r2_score(true: torch.Tensor, pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return per-target R^2 scores."""
    true_2d = _ensure_2d_tensor(true)
    pred_2d = _ensure_2d_tensor(pred)
    residual_sum = ((true_2d - pred_2d) ** 2).sum(dim=0)
    centered_true = true_2d - true_2d.mean(dim=0, keepdim=True)
    total_sum = (centered_true ** 2).sum(dim=0).clamp_min(eps)
    return 1.0 - (residual_sum / total_sum)


def torch_pr2_score(true: torch.Tensor, pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return per-target signed Pearson r^2 scores."""
    true_2d = _ensure_2d_tensor(true)
    pred_2d = _ensure_2d_tensor(pred)
    true_centered = true_2d - true_2d.mean(dim=0, keepdim=True)
    pred_centered = pred_2d - pred_2d.mean(dim=0, keepdim=True)
    numerator = (true_centered * pred_centered).sum(dim=0)
    denominator = (
        torch.sqrt((true_centered ** 2).sum(dim=0))
        * torch.sqrt((pred_centered ** 2).sum(dim=0))
    ).clamp_min(eps)
    r = numerator / denominator
    return torch.sign(r) * (r ** 2)


class TorchRidgeRegressor:
    """Torch-native ridge regressor with sklearn-like API."""

    def __init__(self, alpha: float = 0.0, device: str = "cuda"):
        self.alpha = alpha
        self.device = torch.device(device if device else "cuda")
        self.W: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "TorchRidgeRegressor":
        X = X.to(self.device, dtype=torch.float32)
        y = _ensure_2d_tensor(y).to(self.device, dtype=torch.float32)
        n_features = X.shape[1]
        identity = torch.eye(n_features, device=self.device, dtype=X.dtype)
        gram = X.T @ X + self.alpha * identity
        rhs = X.T @ y
        self.W = torch.linalg.solve(gram, rhs)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            raise RuntimeError("Regressor must be fit before predict.")
        X = X.to(self.device, dtype=torch.float32)
        return X @ self.W

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        y = _ensure_2d_tensor(y).to(self.device, dtype=torch.float32)
        preds = self.predict(X)
        # Match sklearn multi-output behavior: uniform average over outputs.
        return torch_r2_score(y, preds).mean().item()


def get_mean_of_uniform(a: float, b: float) -> float:
    """Compute mean of uniform distribution."""
    return (a + b) / 2


def get_std_of_uniform(a: float, b: float) -> float:
    """Compute standard deviation of uniform distribution."""
    return np.sqrt(np.square(b - a) / 12.0)


def get_std_of_discrete_uniform(a: float, b: float) -> float:
    """Compute standard deviation of discrete uniform distribution."""
    return np.sqrt(np.square(b - a + 1) / 12.0)


def get_uniform_stats(a: float, b: float) -> Tuple[float, float]:
    """Get mean and std of uniform distribution."""
    mean = get_mean_of_uniform(a, b)
    std = get_std_of_uniform(a, b)
    return mean, std


def get_discrete_uniform_stats(a: float, b: float) -> Tuple[float, float]:
    """Get mean and std of discrete uniform distribution."""
    mean = get_mean_of_uniform(a, b)
    std = get_std_of_discrete_uniform(a, b)
    return mean, std


def get_mean_of_loguniform(a: float, b: float) -> float:
    """Compute mean of log-uniform distribution."""
    return (b - a) / (np.log(b) - np.log(a))


def get_std_of_loguniform(a: float, b: float) -> float:
    """Compute standard deviation of log-uniform distribution."""
    log_diff = np.log(b) - np.log(a)
    numerator = log_diff * (np.square(b) - np.square(a)) - (2 * np.square(b - a))
    denominator = 2 * np.square(log_diff)
    var = np.divide(numerator, denominator)
    std = np.sqrt(var)
    return std


def get_loguniform_stats(a: float, b: float) -> Tuple[float, float]:
    """Get mean and std of log-uniform distribution."""
    mean = get_mean_of_loguniform(a, b)
    std = get_std_of_loguniform(a, b)
    return mean, std


# Compute augmentation parameter statistics
db_snr_mean, db_snr_std = get_uniform_stats(*SNR_RANGE)
pitch_mean, pitch_std = get_uniform_stats(*PITCH_RANGE)
tempo_mean, tempo_std = get_uniform_stats(*TEMPO_RANGE)
time_shift_mean, time_shift_std = get_uniform_stats(*TIME_SHIFT_RANGE)

order_mean, order_std = get_discrete_uniform_stats(*FILTER_ORDER_RANGE)
low_cutoff_mean, low_cutoff_std = get_loguniform_stats(*BANDPASS_FREQ_LOW_RANGE)
high_cutoff_mean, high_cutoff_std = get_loguniform_stats(
    *BANDPASS_FREQ_HIGH_RANGE
)

PARAMS_MEAN = np.array(
    [
        db_snr_mean,
        pitch_mean,
        tempo_mean,
        time_shift_mean,
        order_mean,
        low_cutoff_mean,
        high_cutoff_mean,
    ],
    dtype=np.float32,
)
PARAMS_STD = np.array(
    [
        db_snr_std,
        pitch_std,
        tempo_std,
        time_shift_std,
        order_std,
        low_cutoff_std,
        high_cutoff_std,
    ],
    dtype=np.float32,
)


class DBSNRAugmentation:
    """dB SNR augmentation class."""

    def __init__(self, low_db: float = -10, high_db: float = 10):
        """Initialize dB SNR augmentation.

        Args:
            low_db: Lower bound for dB SNR range
            high_db: Upper bound for dB SNR range
        """
        self.torch_set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(
            dbspl=DBSPL_LEVEL, use_np=False
        )
        self.crop = at.CenterCrop(crop_length=CROP_LENGTH)
        self.combine_db_snr = at.CombineWithRandomDBSNRWithParam(low_db, high_db)

    def __call__(
        self, aud: np.ndarray, background: np.ndarray
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply dB SNR augmentation.

        Args:
            aud: Input audio signal
            background: Background noise signal

        Returns:
            Tuple of (clean_audio, augmented_audio) and dB SNR parameter
        """
        logging.getLogger("sox").setLevel(logging.ERROR)

        assert aud is not None, "aud is None on input"
        clean_aud = self.crop(aud)
        assert clean_aud is not None, "clean aud is None post crop"

        clean_aud = torch.from_numpy(clean_aud)
        background = torch.from_numpy(background)
        aug_aud, db_snr = self.combine_db_snr(clean_aud, background)
        aug_aud, _ = self.torch_set_dbSPL(aug_aud, None)
        aug_aud = aug_aud.reshape(1, -1)
        db_snr_param = torch.tensor([db_snr])
        clean_aud, _ = self.torch_set_dbSPL(clean_aud, None)
        clean_aud = clean_aud.reshape(1, -1)
        return (clean_aud, aug_aud), db_snr_param


class FilterAugmentation:
    """Filter augmentation class (pitch, tempo, filter)."""

    def __init__(self, low_db: float = -10, high_db: float = 10):
        """Initialize filter augmentation.

        Args:
            low_db: Lower bound for dB SNR range (unused but kept for compatibility)
            high_db: Upper bound for dB SNR range (unused but kept for compatibility)
        """
        self.np_set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(
            dbspl=DBSPL_LEVEL, use_np=True
        )
        self.torch_set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(
            dbspl=DBSPL_LEVEL, use_np=False
        )
        self.crop = at.CenterCrop(crop_length=CROP_LENGTH)

        self.pitch = at.ApplySingleAugmentSox("pitch", return_params=True)
        self.tempo = at.ApplySingleAugmentSox("tempo", return_params=True)
        self.filter = at.ApplySingleAugmentSox("filter", return_params=True)

    def __call__(
        self, aud: np.ndarray, background: Optional[np.ndarray]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply filter augmentation (pitch, tempo, filter).

        Args:
            aud: Input audio signal
            background: Background noise (unused)

        Returns:
            Tuple of (clean_audio, augmented_audio) and augmentation parameters
        """
        logging.getLogger("sox").setLevel(logging.ERROR)

        assert aud is not None, "aud is None on input"
        clean_aud = self.crop(aud)
        assert clean_aud is not None, "clean aud is None post crop"
        clean_aud, _ = self.np_set_dbSPL(clean_aud, None)

        aug_aud, n_semitones = self.pitch(clean_aud)
        aug_aud, temp_shift = self.tempo(aug_aud)
        aug_aud, (order, low_cutoff, high_cutoff) = self.filter(aug_aud)
        aug_aud = torch.from_numpy(aug_aud)
        aug_aud, _ = self.torch_set_dbSPL(aug_aud, None)
        aug_aud = aug_aud.reshape(1, -1)
        params = torch.tensor([n_semitones, temp_shift, order, low_cutoff, high_cutoff])
        clean_aud = torch.from_numpy(clean_aud).reshape(1, -1)
        return (clean_aud, aug_aud), params


class TimeShiftAugmentation:
    """Time shift augmentation class."""

    def __init__(
        self, min_shift: float = -1.0, max_shift: float = 1.0, sample_rate: int = 20_000
    ):
        """Initialize time shift augmentation.

        Args:
            min_shift: Minimum time shift in seconds
            max_shift: Maximum time shift in seconds
            sample_rate: Sample rate of the audio
        """
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.sample_rate = sample_rate
        self.torch_set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(
            dbspl=DBSPL_LEVEL, use_np=False
        )
        self.crop = at.CenterCrop(crop_length=CROP_LENGTH)

    def __call__(
        self, aud: np.ndarray, background: Optional[np.ndarray]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply time shift augmentation.

        Args:
            aud: Input audio signal
            background: Background noise (unused)

        Returns:
            Tuple of (clean_audio, augmented_audio) and time shift parameter
        """
        logging.getLogger("sox").setLevel(logging.ERROR)

        assert aud is not None, "aud is None on input"
        clean_aud = self.crop(aud)
        assert clean_aud is not None, "clean aud is None post crop"

        clean_aud = torch.from_numpy(clean_aud).float()
        clean_aud, _ = self.torch_set_dbSPL(clean_aud, None)

        time_shift_sec = np.random.uniform(self.min_shift, self.max_shift)
        shift_samples = int(time_shift_sec * self.sample_rate)
        aug_aud = torch.roll(clean_aud, shifts=shift_samples, dims=-1)

        aug_aud = aug_aud.reshape(1, -1)
        clean_aud = clean_aud.reshape(1, -1)
        time_shift_param = torch.tensor([time_shift_sec])

        return (clean_aud, aug_aud), time_shift_param


# Initialize augmentation transforms
db_snr_transform = DBSNRAugmentation()
filter_transform = FilterAugmentation()
time_shift_transform = TimeShiftAugmentation()


def _convert_labels_to_tensor(labels):
    """Convert labels to torch tensors."""
    if isinstance(labels, dict):
        return {k: torch.from_numpy(v) for k, v in labels.items()}
    return torch.from_numpy(labels)


def _create_collate_fn(transform):
    """Create a collate function for a given augmentation transform."""

    def collate_fn(batch):
        """Collate function for data loader."""
        batch = batch[0]  # Unbox wrapper added by dataloader
        clean_signals = []
        augmented_signals = []
        params = []
        labels = batch[-1]  # Labels already collated
        labels = _convert_labels_to_tensor(labels)

        # Process signal and noise pairs
        signal_noise_pairs = zip(*batch[:2])
        for signal, noise in signal_noise_pairs:
            if signal is None or signal.sum() == 0:
                continue

            (clean_sig, augment_sig), params_i = transform(signal, noise)
            clean_signals.append(clean_sig)
            augmented_signals.append(augment_sig)
            params.append(params_i)

        clean_signals = torch.cat(clean_signals).unsqueeze(1).float()
        augmented_signals = torch.cat(augmented_signals).unsqueeze(1).float()
        params = torch.stack(params)
        return clean_signals, augmented_signals, params, labels

    return collate_fn


snr_collate_fn = _create_collate_fn(db_snr_transform)
filter_collate_fn = _create_collate_fn(filter_transform)
time_shift_collate_fn = _create_collate_fn(time_shift_transform)


@dataclass
class AugmentationConfig:
    """Configuration for an augmentation type."""
    name: str
    collate_fn: Callable
    param_names: List[str]
    param_indices: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.param_indices:
            self.param_indices = list(range(len(self.param_names)))


# Define augmentation configurations
AUGMENTATION_CONFIGS = [
    AugmentationConfig(
        name="snr",
        collate_fn=snr_collate_fn,
        param_names=["dB SNR"],
        param_indices=[0],
    ),
    AugmentationConfig(
        name="filter",
        collate_fn=filter_collate_fn,
        param_names=["Pitch (semitones)", "% Time warp", "Filter order", 
                     "Filter low cutoff", "Filter high cutoff"],
        param_indices=[0, 1, 2, 3, 4],
    ),
    AugmentationConfig(
        name="time_shift",
        collate_fn=time_shift_collate_fn,
        param_names=["Time shift (s)"],
        param_indices=[0],
    ),
]


def _estimate_whisper_valid_tokens(
    input_tensor: torch.Tensor,
    waveform_sr: int,
    max_tokens: int = 1500,
) -> int:
    """Estimate number of non-padded Whisper encoder tokens from input duration."""
    n_samples = int(input_tensor.shape[-1])
    # Whisper encoder context is 30 s -> 1500 tokens, i.e. 50 tokens/s.
    token_rate_hz = max_tokens / 30.0
    duration_seconds = n_samples / float(waveform_sr)
    n_tokens = int(round(duration_seconds * token_rate_hz))
    return max(1, min(max_tokens, n_tokens))


def get_rep_wrapped_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    layer: str,
    waveform_sr: Optional[int] = None,
) -> torch.Tensor:
    """Extract representation from model at specified layer.

    Args:
        model: Model to extract representations from
        input_tensor: Input tensor
        layer: Layer name to extract from
        waveform_sr: Sample rate of ``input_tensor`` when ``model`` is AudioMAE
            (e.g. JSIN at 20 kHz). Ignored for Kell/SSL models.

    Returns:
        Flattened representation tensor
    """
    if isinstance(model, AudioMAELayerwiseEncoder):
        sr_use = waveform_sr if waveform_sr is not None else AUDIOMAE_SR
        embeddings = model(input_tensor, sr=sr_use)
        rep = embeddings[layer]
        return rep.flatten(start_dim=1) if rep.dim() > 2 else rep
    if isinstance(model, WhisperLayerwiseEncoder):
        sr_use = waveform_sr if waveform_sr is not None else SAMPLE_RATE
        embeddings = model(input_tensor, sr=sr_use, flatten_activations=False)
        rep = embeddings[layer]
        # Whisper pads/trims to 30 s internally; keep only valid, unpadded tokens.
        if rep.dim() != 3:
            raise ValueError(
                f"Expected unflattened Whisper activations [B, T, D], got {rep.shape}"
            )
        valid_tokens = _estimate_whisper_valid_tokens(input_tensor, waveform_sr=sr_use)
        rep = rep[:, :valid_tokens, :]
        return rep.flatten(start_dim=1) if rep.dim() > 2 else rep

    if layer == "invar_head":
        feature, rep, logits = model(input_tensor)
        if len(rep) == 2:
            rep, _ = rep
    elif layer == "equivar_head":
        feature, (_, rep), logits = model(input_tensor)
    else:
        predictions, rep, all_outputs = model(
            input_tensor, with_latent=True, fake_relu=False
        )
        rep = all_outputs[layer]
    rep = rep.flatten(start_dim=1)
    return rep


def extract_features_param_decoding(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    layer: str = "avgpool",
    num_batches: Optional[int] = None,
    waveform_sr: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Extract features for parameter decoding.

    Args:
        model: Model to extract features from
        loader: Data loader
        layer: Layer to extract representations from
        num_batches: Number of batches to process (None for all)
        waveform_sr: Input waveform sample rate for AudioMAE (ignored otherwise)

    Returns:
        Tuple of (clean_responses, augmented_responses, params, None)
    """
    if num_batches is None or num_batches == -1:
        num_batches = len(loader)

    responses_clean, responses_augmented = [], []
    params, labels = [], []
    n_processed = 0

    with torch.no_grad():
        for clean_audio, augmented_audio, param, label in tqdm(loader):
            clean_audio = clean_audio.cuda()
            augmented_audio = augmented_audio.cuda()
            responses_clean.append(
                get_rep_wrapped_model(
                    model, clean_audio, layer, waveform_sr=waveform_sr
                ).cpu()
            )
            responses_augmented.append(
                get_rep_wrapped_model(
                    model, augmented_audio, layer, waveform_sr=waveform_sr
                ).cpu()
            )

            params.append(param)
            labels.append(label)
            n_processed += 1
            if n_processed == num_batches:
                break

    responses_clean = torch.cat(responses_clean)
    responses_augmented = torch.cat(responses_augmented)
    params = torch.cat(params)

    return responses_clean, responses_augmented, params, None


def load_model(
    config_path: Path,
    checkpoint_path: Optional[str],
    exp_dir: Path,
    is_supervised: bool,
) -> Tuple[torch.nn.Module, List[str]]:
    """Load model from checkpoint.

    Args:
        config_path: Path to model config file
        checkpoint_path: Path to checkpoint (None to use latest)
        exp_dir: Experiment directory
        is_supervised: Whether model is supervised (can be overridden by config check)

    Returns:
        Tuple of (model, available_layers)
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Check config file to determine if it's supervised
    # Supervised models use 'arch_params' and 'kell2018_multi_task' architecture
    # SSL models use 'arch_kwargs' and SSL architectures (names starting with 'SSL')
    arch_name = config.get("model", {}).get("arch_name", "")
    
    # If architecture starts with 'SSL', it's definitely an SSL model
    is_ssl_arch = arch_name.startswith("SSL") if arch_name else False
    
    # Supervised models use 'arch_params' and 'kell2018_multi_task' architecture
    # But exclude SSL architectures even if they're in supervised_models directory
    config_is_supervised = (
        not is_ssl_arch
        and (
            "arch_params" in config.get("model", {})
            or arch_name == "kell2018_multi_task"
            or ("supervised_models" in str(config_path) and not is_ssl_arch)
        )
    )
    
    # Use explicit flag if provided, otherwise use config-based detection
    # But explicit flag can't override SSL architecture detection
    if is_ssl_arch:
        is_supervised = False
    else:
        is_supervised = is_supervised or config_is_supervised

    if checkpoint_path == "" or checkpoint_path is None:
        checkpoint_dir = exp_dir / f"{config_path.stem}/checkpoints"
        checkpoint_path = str(sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)[-1])

    print(f"Model checkpoint path: {checkpoint_path}")
    print(f"Loading as {'supervised' if is_supervised else 'SSL'} model")

    if is_supervised:
        model = LitWordAudioSetModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, config=config
        )
        all_layers = model.metamer_layers
        model = model.model.eval().cuda()
    else:
        model = LitAudioSSL.load_from_checkpoint(
            config=config, checkpoint_path=checkpoint_path
        )
        all_layers = model.metamer_layers
        model = model.model.eval().cuda()

    return model, all_layers


def save_plots(
    r2_scores: Dict[str, float],
    pr2_scores: Dict[str, float],
    model_name: str,
    layer: str,
    ridge_alpha: float,
    output_dir: Path,
):
    """Save visualization plots.

    Args:
        r2_scores: R^2 scores dictionary
        pr2_scores: Pearson's r^2 scores dictionary
        model_name: Name of the model
        layer: Layer name
        ridge_alpha: Ridge regression alpha parameter
        output_dir: Output directory for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ridge_title_str = f"Ridge $\\alpha=${ridge_alpha:.1f}" if ridge_alpha != 0 else ""

    # R^2 plot
    plt.figure(figsize=(12, 5))
    bar_x = np.arange(len(r2_scores))
    plt.bar(bar_x, list(r2_scores.values()), color="blue", alpha=0.5, width=0.5)
    plt.axhline(0, color="k", lw=0.5)
    plt.xticks(rotation=45, ticks=bar_x, labels=list(r2_scores.keys()))
    plt.xlabel("Augmentation")
    plt.ylabel("$R^2$ Score")
    plt.ylim(-1.1, 1.1)
    plt.title(f"Model: {model_name}\nLayer: {layer}\n{ridge_title_str}")
    fig_name = (
        f"{layer}_param_decoding_r2_by_augmentation"
        f"{f'_{ridge_alpha:.0e}' if ridge_alpha != 0.0 else ''}"
    )
    plt.savefig(output_dir / f"{fig_name}.png", transparent=False, bbox_inches="tight")
    plt.close()

    # Pearson's r^2 plot
    plt.figure(figsize=(12, 5))
    bar_x = np.arange(len(pr2_scores))
    plt.bar(bar_x, list(pr2_scores.values()), color="blue", alpha=0.5, width=0.5)
    plt.axhline(0, color="k", lw=0.5)
    plt.xticks(rotation=45, ticks=bar_x, labels=list(pr2_scores.keys()))
    plt.xlabel("Augmentation")
    plt.ylabel("Pearson's $r^2$ Score")
    plt.ylim(-1.1, 1.1)
    plt.title(f"Model: {model_name}\nLayer: {layer}\n{ridge_title_str}")
    fig_name = (
        f"{layer}_param_decoding_pearsons_r2_by_augmentation"
        f"{f'_{ridge_alpha:.0e}' if ridge_alpha != 0.0 else ''}"
    )
    plt.savefig(output_dir / f"{fig_name}.png", transparent=False, bbox_inches="tight")
    plt.close()


class ParameterDecodingEvaluator:
    """Evaluator for parameter decoding from model representations.
    
    This class encapsulates the feature extraction and regression pipeline
    for evaluating how well augmentation parameters can be decoded from
    model representations.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        layer: str,
        ridge_alpha: float = 0.0,
        num_workers: int = 1,
        waveform_sr: Optional[int] = None,
        regression_device: Optional[str] = None,
    ):
        """Initialize the evaluator.
        
        Args:
            model: Model to extract representations from
            layer: Layer name to extract representations from
            ridge_alpha: Ridge regularization parameter
            num_workers: Number of data loader workers
            waveform_sr: Sample rate for AudioMAE inputs (None for Kell/SSL models)
            regression_device: Device for ridge regression operations
        """
        self.model = model
        self.layer = layer
        self.ridge_alpha = ridge_alpha
        self.num_workers = num_workers
        self.waveform_sr = waveform_sr
        self.loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.regression_device = regression_device or default_device

    def extract_features(
        self,
        loader: torch.utils.data.DataLoader,
        num_batches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from the model for a given data loader.
        
        Args:
            loader: Data loader with augmented samples
            num_batches: Number of batches to process
            
        Returns:
            Tuple of (clean_responses, augmented_responses, params)
        """
        rc, ra, params, _ = extract_features_param_decoding(
            self.model,
            loader,
            layer=self.layer,
            num_batches=num_batches,
            waveform_sr=self.waveform_sr,
        )
        return rc, ra, params
    
    def prepare_regression_data(
        self,
        rc_train: torch.Tensor,
        ra_train: torch.Tensor,
        params_train: torch.Tensor,
        rc_test: torch.Tensor,
        ra_test: torch.Tensor,
        params_test: torch.Tensor,
        aug_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare extracted features for regression.
        
        Combines clean and augmented responses, filters invalid samples,
        and normalizes targets.
        
        Args:
            rc_train: Clean responses for training
            ra_train: Augmented responses for training
            params_train: Parameters for training
            rc_test: Clean responses for testing
            ra_test: Augmented responses for testing
            params_test: Parameters for testing
            aug_name: Augmentation name (used for SNR filtering)
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test)
        """
        # Combine features: concatenate clean and augmented
        X_train = torch.cat([rc_train, ra_train], dim=1).detach().float()
        X_test = torch.cat([rc_test, ra_test], dim=1).detach().float()
        
        # Prepare targets
        Y_train = params_train.detach().float()
        Y_test = params_test.detach().float()
        
        # Ensure 2D shape
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)
            Y_test = Y_test.reshape(-1, 1)
        
        # Filter out invalid SNR examples
        if aug_name == "snr":
            train_valid = ~torch.isinf(Y_train).any(dim=1)
            test_valid = ~torch.isinf(Y_test).any(dim=1)
            X_train = X_train[train_valid]
            X_test = X_test[test_valid]
            Y_train = Y_train[train_valid]
            Y_test = Y_test[test_valid]
        
        # Normalize parameters using training stats
        mean, std = Y_train.mean(dim=0), Y_train.std(dim=0).clamp_min(1e-12)
        Y_train = (Y_train - mean) / std
        Y_test = (Y_test - mean) / std
        
        return X_train, X_test, Y_train, Y_test
    
    def fit_regression(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
    ) -> TorchRidgeRegressor:
        """Fit ridge regression model.
        
        Args:
            X_train: Training features
            Y_train: Training targets
            
        Returns:
            Fitted ridge regression model
        """
        with torch.no_grad():
            regression = TorchRidgeRegressor(
                alpha=self.ridge_alpha, device=self.regression_device
            )
            return regression.fit(X_train, Y_train)
    
    def evaluate_regression(
        self,
        regression: TorchRidgeRegressor,
        X_test: torch.Tensor,
        Y_test: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Evaluate regression model on test data.
        
        Args:
            regression: Fitted regression model
            X_test: Test features
            Y_test: Test targets
            
        Returns:
            Tuple of (predictions, R^2 score)
        """
        with torch.no_grad():
            preds = regression.predict(X_test)
            score = regression.score(X_test, Y_test)
        return preds, score
    
    def compute_per_param_scores(
        self,
        Y_test: torch.Tensor,
        preds: torch.Tensor,
        param_names: List[str],
        score_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Dict[str, float]:
        """Compute scores for each parameter.
        
        Args:
            Y_test: True test values
            preds: Predicted values
            param_names: Names of parameters
            score_func: Scoring function
            
        Returns:
            Dictionary mapping parameter names to scores
        """
        scores: Dict[str, float] = {}
        Y_2d = _ensure_2d_tensor(Y_test)
        P_2d = _ensure_2d_tensor(preds)
        for idx, param_name in enumerate(param_names):
            true_vals = Y_2d[:, idx]
            pred_vals = P_2d[:, idx]
            scores[param_name] = score_func(true_vals, pred_vals).item()
        return scores
    
    def evaluate_augmentation(
        self,
        train_dataset,
        val_dataset,
        aug_config: AugmentationConfig,
        num_train: int,
        num_eval: int,
        seed: int = 0,
    ) -> Dict:
        """Evaluate parameter decoding for a single augmentation type.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            aug_config: Augmentation configuration
            num_train: Number of training batches
            num_eval: Number of evaluation batches
            seed: Random seed
            
        Returns:
            Dictionary with evaluation results
        """
        seed_everything(seed)
        
        # Create loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, collate_fn=aug_config.collate_fn, **self.loader_kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            val_dataset, collate_fn=aug_config.collate_fn, **self.loader_kwargs
        )
        
        # Extract features
        rc_train, ra_train, params_train = self.extract_features(train_loader, num_train)
        rc_test, ra_test, params_test = self.extract_features(test_loader, num_eval)
        
        # Prepare data
        X_train, X_test, Y_train, Y_test = self.prepare_regression_data(
            rc_train, ra_train, params_train,
            rc_test, ra_test, params_test,
            aug_config.name,
        )
        
        # Fit and evaluate regression
        regression = self.fit_regression(X_train, Y_train)
        preds, overall_score = self.evaluate_regression(regression, X_test, Y_test)
        
        # Compute per-parameter scores
        r2_scores = self.compute_per_param_scores(
            Y_test=Y_test,
            preds=preds,
            param_names=aug_config.param_names,
            score_func=lambda y, p: torch_r2_score(y, p).squeeze()
        )
        pr2_scores = self.compute_per_param_scores(
            Y_test=Y_test,
            preds=preds,
            param_names=aug_config.param_names,
            score_func=lambda y, p: torch_pr2_score(y, p).squeeze()
        )
        
        return {
            "overall_score": overall_score,
            "r2_scores": r2_scores,
            "pr2_scores": pr2_scores,
        }
    
    def run_evaluation(
        self,
        train_dataset,
        val_dataset,
        num_train: int,
        num_eval: int,
        n_runs: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """Run full evaluation across all augmentations and multiple runs.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_train: Number of training batches
            num_eval: Number of evaluation batches
            n_runs: Number of runs per augmentation
            verbose: Whether to print progress
            
        Returns:
            Dictionary with all results and summary statistics
        """
        all_run_results = {
            "r2_scores": [],
            "pr2_scores": [],
            "per_aug_r2": {},
        }
        
        for run_idx in range(n_runs):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_runs} ===")
            
            run_r2_scores = {}
            run_pr2_scores = {}
            
            for aug_config in AUGMENTATION_CONFIGS:
                if verbose:
                    print(f"\nProcessing augmentation: {aug_config.name}")
                
                result = self.evaluate_augmentation(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    aug_config=aug_config,
                    num_train=num_train,
                    num_eval=num_eval,
                    seed=run_idx,
                )
                
                if verbose:
                    print(f"  Overall R^2: {result['overall_score']:.3f}")
                    for param_name, score in result["r2_scores"].items():
                        print(f"  {param_name}: R^2={score:.3f}")
                
                # Store results
                if aug_config.name not in all_run_results["per_aug_r2"]:
                    all_run_results["per_aug_r2"][aug_config.name] = []
                all_run_results["per_aug_r2"][aug_config.name].append(result["overall_score"])
                
                run_r2_scores.update(result["r2_scores"])
                run_pr2_scores.update(result["pr2_scores"])
            
            all_run_results["r2_scores"].append(run_r2_scores)
            all_run_results["pr2_scores"].append(run_pr2_scores)
        
        # Compute summary statistics
        summary = self._compute_summary_statistics(all_run_results, verbose)
        
        return {
            "r2_scores": summary["r2_summary"],
            "pr2_scores": summary["pr2_summary"],
            "per_aug_r2": all_run_results["per_aug_r2"],
            "n_runs": n_runs,
            "all_runs": all_run_results,
        }
    
    def _compute_summary_statistics(
        self,
        all_run_results: Dict,
        verbose: bool = True,
    ) -> Dict:
        """Compute summary statistics across runs.
        
        Args:
            all_run_results: Results from all runs
            verbose: Whether to print summary
            
        Returns:
            Dictionary with r2 and pr2 summaries
        """
        if verbose:
            print("\n=== Summary Statistics ===")
        
        all_param_names = []
        for aug_config in AUGMENTATION_CONFIGS:
            all_param_names.extend(aug_config.param_names)
        
        r2_summary = {}
        pr2_summary = {}
        
        for param_name in all_param_names:
            r2_values = [run[param_name] for run in all_run_results["r2_scores"]]
            pr2_values = [run[param_name] for run in all_run_results["pr2_scores"]]
            
            r2_summary[param_name] = {
                "mean": np.mean(r2_values),
                "std": np.std(r2_values),
                "values": r2_values,
            }
            pr2_summary[param_name] = {
                "mean": np.mean(pr2_values),
                "std": np.std(pr2_values),
                "values": pr2_values,
            }
            
            if verbose:
                print(f"{param_name}: R^2 = {r2_summary[param_name]['mean']:.3f} "
                      f"± {r2_summary[param_name]['std']:.3f}")
        
        return {"r2_summary": r2_summary, "pr2_summary": pr2_summary}


def main(args):
    """Main function to run parameter decoding evaluation."""
    # Create datasets
    train_dataset = jsinV3_precombined_all_signals(
        root=JSIN_PATH, train=True, transform=None, batch_size=args.batch_size
    )
    val_dataset = jsinV3_precombined_all_signals(
        root=JSIN_PATH,
        train=False,
        transform=None,
        batch_size=args.batch_size,
        eval_max=5,
    )

    waveform_sr: Optional[int] = None

    if args.model_type == "audiomae":
        print("Loading pretrained AudioMAE (hance-ai/audiomae)")
        model = AudioMAELayerwiseEncoder(time_pool=args.audiomae_time_pool).cuda().eval()
        all_layers = list(model.layer_names)
        model_name = args.model_name or "audiomae_pretrained"
        waveform_sr = args.input_sample_rate
        print(f"Running model: {model_name} (AudioMAE, {len(all_layers)} layers)")
    elif args.model_type == "whisper":
        print(f"Loading pretrained Whisper ({args.whisper_model})")
        model = WhisperLayerwiseEncoder(whisper_model_name=args.whisper_model).cuda().eval()
        all_layers = list(model.layer_names)
        model_name = args.model_name or f"whisper_pretrained_{args.whisper_model}"
        waveform_sr = args.input_sample_rate
        print(f"Running model: {model_name} (Whisper, {len(all_layers)} layers)")
    else:
        config_path = Path(args.model_config)
        model_name = config_path.stem
        is_supervised = args.supervised or any(
            keyword in model_name for keyword in ["supervised", "audioset"]
        )

        print(f"Running model: {model_name} (supervised={is_supervised})")

        model, all_layers = load_model(
            config_path, args.model_ckpt, args.exp_dir, is_supervised
        )

    if args.job_id > -1:
        if args.job_id >= len(all_layers):
            raise ValueError(
                f"job_id {args.job_id} out of range; model has {len(all_layers)} layers: "
                f"{all_layers}"
            )
        layer = all_layers[args.job_id]
    else:
        layer = args.layer
        if isinstance(model, AudioMAELayerwiseEncoder):
            layer = parse_audiomae_layer_str(layer, valid_layers=all_layers)
        elif isinstance(model, WhisperLayerwiseEncoder):
            layer = parse_whisper_layer_str(layer, valid_layers=all_layers)

    print(f"Layer: {layer}")
    print(f"Running {args.n_runs} regression runs per augmentation")

    # Create evaluator and run evaluation
    evaluator = ParameterDecodingEvaluator(
        model=model,
        layer=layer,
        ridge_alpha=args.ridge_alpha,
        num_workers=args.num_workers,
        waveform_sr=waveform_sr,
        regression_device=args.regression_device,
    )
    
    results = evaluator.run_evaluation(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_train=args.num_train,
        num_eval=args.num_eval,
        n_runs=args.n_runs,
        verbose=True,
    )

    # Save plots and results
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_plots:
        mean_r2 = {k: v["mean"] for k, v in results["r2_scores"].items()}
        mean_pr2 = {k: v["mean"] for k, v in results["pr2_scores"].items()}
        save_plots(mean_r2, mean_pr2, model_name, layer, args.ridge_alpha, output_dir)

    data_out_name = (
        output_dir
        / f"{layer}_r2_decoding_values"
        f"{f'_ridge_alpha_{args.ridge_alpha:.0e}' if args.ridge_alpha != 0.0 else ''}"
        f"_n_runs_{args.n_runs}.pkl"
    )

    with open(data_out_name, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nResults saved to: {data_out_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=192,
        type=int,
        help="Batch size used to extract representations",
    )
    parser.add_argument(
        "--num_train",
        default=50,
        type=int,
        help="Number of training batches to take from training set",
    )
    parser.add_argument(
        "--num_eval",
        default=5,
        type=int,
        help="Number of batches to take from validation set",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers per dataloader",
    )
    parser.add_argument(
        "--layer",
        default="avgpool",
        type=str,
        help="Layer to extract representations from",
    )
    parser.add_argument(
        "--job_id",
        default=-1,
        type=int,
        help="Slurm job array index, used to select layers",
    )
    parser.add_argument(
        "--exp_dir",
        default=Path("./model_checkpoints"),
        type=Path,
        help="Directory to save checkpoints and logs to",
    )
    parser.add_argument(
        "--model_config",
        default=Path(
            "model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_5e-01_audioset_only.yaml"
        ),
        type=Path,
        help="Path to model config",
    )
    parser.add_argument(
        "--model_ckpt",
        default="",
        type=str,
        help="Path to model checkpoint. If empty, uses latest checkpoint from exp_dir",
    )
    parser.add_argument(
        "--supervised",
        action="store_true",
        help="Force model to be treated as supervised model. Otherwise auto-detects from config name",
    )
    parser.add_argument(
        "--ridge_alpha",
        default=0.0,
        type=float,
        help="Alpha to use in ridge regression. Default (0) is same as OLS",
    )
    parser.add_argument(
        "--regression_device",
        default=None,
        type=str,
        help="Device for torch ridge regression (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--n_runs",
        default=5,
        type=int,
        help="Number of regression runs per augmentation (default: 5)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Generate and save plots (disabled by default)",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("parameter_decoding"),
        type=Path,
        help="Directory to save plots and results to",
    )
    parser.add_argument(
        "--model_type",
        default="from_config",
        choices=["from_config", "audiomae", "whisper"],
        help=(
            "from_config: YAML + checkpoint (default); "
            "audiomae: pretrained AudioMAE encoder; "
            "whisper: pretrained Whisper encoder"
        ),
    )
    parser.add_argument(
        "--audiomae_time_pool",
        action="store_true",
        help="Average over time dimension in AudioMAE representations (smaller vectors)",
    )
    parser.add_argument(
        "--input_sample_rate",
        default=SAMPLE_RATE,
        type=int,
        help="Waveform sample rate for JSIN audio when using AudioMAE (default: JSIN 20 kHz)",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "Output subdirectory name for pretrained encoder modes "
            "(default: audiomae_pretrained or whisper_pretrained_<model>)"
        ),
    )
    parser.add_argument(
        "--whisper_model",
        default="large-v3",
        type=str,
        help="Pretrained Whisper model name when --model_type whisper",
    )
    args = parser.parse_args()
    main(args)
