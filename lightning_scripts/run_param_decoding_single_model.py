"""Parameter decoding evaluation script for single model evaluation.

This script evaluates a single model's ability to decode augmentation parameters
from its representations. It supports both SSL and supervised models.
"""

import logging
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import robustness.audio_functions.audio_transforms as at
import torch
import yaml
from lightning import seed_everything
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm

from lightning_scripts.jsinV3DataLoader_precombined_batched import (
    jsinV3_precombined_all_signals,
)
from lightning_scripts.lightning_classifier_matched_speech_in_noise import (
    LitWordAudioSetModel,
)
from lightning_scripts.lightning_ssl_matched_speech_in_noise import LitAudioSSL

# Configure PyTorch
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Constants
JSIN_PATH = "/mnt/ceph/users/jfeather/data/training_datasets_audio/JSIN_all_v3/subsets/"
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

AUGMENTATION_LIST = [
    "dB SNR",
    "Pitch (semitones)",
    "% Time warp",
    "Time shift (s)",
    "Filter order",
    "Filter low cutoff",
    "Filter high cutoff",
]


def pearsonr_vec(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation coefficient."""
    return np.vectorize(pearsonr, signature="(n),(n)->(),()")(true, pred)


def pr2_score(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute signed Pearson's r^2 score.

    Args:
        true: True values
        pred: Predicted values

    Returns:
        Signed r^2 score (preserves sign of correlation)
    """
    r, _ = pearsonr_vec(true, pred)
    signs = np.sign(r)
    return signs * np.power(r, 2)


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


def get_rep_wrapped_model(
    model: torch.nn.Module, input_tensor: torch.Tensor, layer: str
) -> torch.Tensor:
    """Extract representation from model at specified layer.

    Args:
        model: Model to extract representations from
        input_tensor: Input tensor
        layer: Layer name to extract from

    Returns:
        Flattened representation tensor
    """
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Extract features for parameter decoding.

    Args:
        model: Model to extract features from
        loader: Data loader
        layer: Layer to extract representations from
        num_batches: Number of batches to process (None for all)

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
                get_rep_wrapped_model(model, clean_audio, layer).cpu()
            )
            responses_augmented.append(
                get_rep_wrapped_model(model, augmented_audio, layer).cpu()
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


def create_data_loaders(
    train_dataset, val_dataset, num_workers: int
) -> Dict[str, torch.utils.data.DataLoader]:
    """Create data loaders for all augmentation types.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_workers: Number of worker processes

    Returns:
        Dictionary of data loaders
    """
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    loaders = {
        "snr_train": torch.utils.data.DataLoader(
            train_dataset, collate_fn=snr_collate_fn, **loader_kwargs
        ),
        "filter_train": torch.utils.data.DataLoader(
            train_dataset, collate_fn=filter_collate_fn, **loader_kwargs
        ),
        "time_shift_train": torch.utils.data.DataLoader(
            train_dataset, collate_fn=time_shift_collate_fn, **loader_kwargs
        ),
        "snr_test": torch.utils.data.DataLoader(
            val_dataset, collate_fn=snr_collate_fn, **loader_kwargs
        ),
        "filter_test": torch.utils.data.DataLoader(
            val_dataset, collate_fn=filter_collate_fn, **loader_kwargs
        ),
        "time_shift_test": torch.utils.data.DataLoader(
            val_dataset, collate_fn=time_shift_collate_fn, **loader_kwargs
        ),
    }
    return loaders


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


def compute_scores(
    y_true: Dict[str, np.ndarray], y_pred: Dict[str, np.ndarray], score_func
) -> Dict[str, float]:
    """Compute scores for each augmentation type.

    Args:
        y_true: True values dictionary
        y_pred: Predicted values dictionary
        score_func: Scoring function (r2_score or pr2_score)

    Returns:
        Dictionary of scores per augmentation
    """
    scores = {}
    for idx, aug_name in enumerate(AUGMENTATION_LIST):
        if idx == 0:  # dB SNR
            true_vals = y_true["snr"][:, 0] if y_true["snr"].ndim > 1 else y_true["snr"]
            pred_vals = y_pred["snr"][:, 0] if y_pred["snr"].ndim > 1 else y_pred["snr"]
            scores[aug_name] = score_func(true_vals, pred_vals)
        elif idx == 3:  # Time shift
            true_vals = y_true["ts"][:, 0] if y_true["ts"].ndim > 1 else y_true["ts"]
            pred_vals = y_pred["ts"][:, 0] if y_pred["ts"].ndim > 1 else y_pred["ts"]
            scores[aug_name] = score_func(true_vals, pred_vals)
        else:  # Filter params
            filter_idx = idx - 1 if idx < 3 else idx - 2
            true_vals = (
                y_true["filter"][:, filter_idx]
                if y_true["filter"].ndim > 1
                else y_true["filter"]
            )
            pred_vals = (
                y_pred["filter"][:, filter_idx]
                if y_pred["filter"].ndim > 1
                else y_pred["filter"]
            )
            scores[aug_name] = score_func(true_vals, pred_vals)
    return scores


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

    # Create data loaders
    loaders = create_data_loaders(train_dataset, val_dataset, args.num_workers)

    # Load model
    config_path = Path(args.model_config)
    model_name = config_path.stem
    is_supervised = args.supervised or any(
        keyword in model_name for keyword in ["supervised", "audioset"]
    )

    print(f"Running model: {model_name} (supervised={is_supervised})")

    model, all_layers = load_model(
        config_path, args.model_ckpt, args.exp_dir, is_supervised
    )

    layer = all_layers[args.job_id] if args.job_id > -1 else args.layer
    print(f"Layer: {layer}")

    # Extract features for each augmentation type
    seed_everything(0)
    (
        rc_train_snr,
        ra_train_snr,
        params_train_snr,
        _,
    ) = extract_features_param_decoding(
        model, loaders["snr_train"], layer=layer, num_batches=args.num_train
    )
    (
        rc_test_snr,
        ra_test_snr,
        params_test_snr,
        _,
    ) = extract_features_param_decoding(
        model, loaders["snr_test"], layer=layer, num_batches=args.num_eval
    )

    seed_everything(0)
    (
        rc_train_filter,
        ra_train_filter,
        params_train_filter,
        _,
    ) = extract_features_param_decoding(
        model, loaders["filter_train"], layer=layer, num_batches=args.num_train
    )
    (
        rc_test_filter,
        ra_test_filter,
        params_test_filter,
        _,
    ) = extract_features_param_decoding(
        model, loaders["filter_test"], layer=layer, num_batches=args.num_eval
    )

    seed_everything(0)
    (
        rc_train_ts,
        ra_train_ts,
        params_train_ts,
        _,
    ) = extract_features_param_decoding(
        model,
        loaders["time_shift_train"],
        layer=layer,
        num_batches=args.num_train,
    )
    (
        rc_test_ts,
        ra_test_ts,
        params_test_ts,
        _,
    ) = extract_features_param_decoding(
        model, loaders["time_shift_test"], layer=layer, num_batches=args.num_eval
    )

    # Combine features: concatenate clean and augmented
    X_train_snr = torch.cat([rc_train_snr, ra_train_snr], dim=1).detach().cpu().numpy()
    X_test_snr = torch.cat([rc_test_snr, ra_test_snr], dim=1).detach().cpu().numpy()
    X_train_filter = torch.cat([rc_train_filter, ra_train_filter], dim=1).detach().cpu().numpy()
    X_test_filter = torch.cat([rc_test_filter, ra_test_filter], dim=1).detach().cpu().numpy()
    X_train_ts = torch.cat([rc_train_ts, ra_train_ts], dim=1).detach().cpu().numpy()
    X_test_ts = torch.cat([rc_test_ts, ra_test_ts], dim=1).detach().cpu().numpy()

    X_test_snr = X_test_snr.reshape(X_test_snr.shape[0], -1)
    X_test_filter = X_test_filter.reshape(X_test_filter.shape[0], -1)
    X_test_ts = X_test_ts.reshape(X_test_ts.shape[0], -1)

    # Prepare targets
    Y_train_snr = params_train_snr[:, 0].detach().cpu().numpy()
    Y_test_snr = params_test_snr[:, 0].detach().cpu().numpy()
    Y_train_filter = params_train_filter.detach().cpu().numpy()
    Y_test_filter = params_test_filter.detach().cpu().numpy()
    Y_train_ts = params_train_ts[:, 0].detach().cpu().numpy()
    Y_test_ts = params_test_ts[:, 0].detach().cpu().numpy()

    # Filter out invalid SNR examples
    train_ixs_snr = np.argwhere(~np.isinf(Y_train_snr)).flatten()
    test_ixs_snr = np.argwhere(~np.isinf(Y_test_snr)).flatten()

    X_train_snr = X_train_snr[train_ixs_snr]
    X_test_snr = X_test_snr[test_ixs_snr]
    Y_train_snr = Y_train_snr[train_ixs_snr].reshape(-1, 1)
    Y_test_snr = Y_test_snr[test_ixs_snr].reshape(-1, 1)

    # Normalize parameters using empirical stats
    snr_mean, snr_std = Y_train_snr.mean(0), Y_train_snr.std(0)
    Y_train_snr = (Y_train_snr - snr_mean) / snr_std
    Y_test_snr = (Y_test_snr - snr_mean) / snr_std

    filter_mean, filter_std = Y_train_filter.mean(0), Y_train_filter.std(0)
    Y_train_filter = (Y_train_filter - filter_mean) / filter_std
    Y_test_filter = (Y_test_filter - filter_mean) / filter_std

    ts_mean, ts_std = Y_train_ts.mean(0), Y_train_ts.std(0)
    Y_train_ts = (Y_train_ts - ts_mean) / ts_std
    Y_test_ts = (Y_test_ts - ts_mean) / ts_std
    
    # Reshape to 2D for consistency with predictions
    Y_train_ts = Y_train_ts.reshape(-1, 1)
    Y_test_ts = Y_test_ts.reshape(-1, 1)

    # Fit regressions
    regression_snr = Ridge(alpha=args.ridge_alpha).fit(X_train_snr, Y_train_snr)
    regression_filter = Ridge(alpha=args.ridge_alpha).fit(X_train_filter, Y_train_filter)
    regression_ts = Ridge(alpha=args.ridge_alpha).fit(X_train_ts, Y_train_ts)

    score_snr = regression_snr.score(X_test_snr, Y_test_snr)
    score_filter = regression_filter.score(X_test_filter, Y_test_filter)
    score_ts = regression_ts.score(X_test_ts, Y_test_ts)

    preds_snr = regression_snr.predict(X_test_snr).reshape(-1, 1)
    preds_filter = regression_filter.predict(X_test_filter)
    preds_ts = regression_ts.predict(X_test_ts).reshape(-1, 1)

    print(f"Model Scores:")
    print(f"{score_snr:.3f} (R^2 dB SNR)")
    print(f"{score_filter:.3f} (R^2 filter)")
    print(f"{score_ts:.3f} (R^2 time shift)")

    # Compute scores per augmentation
    y_true = {"snr": Y_test_snr, "filter": Y_test_filter, "ts": Y_test_ts}
    y_pred = {"snr": preds_snr, "filter": preds_filter, "ts": preds_ts}

    r2_scores = compute_scores(y_true, y_pred, r2_score)
    r2_pr2_scores = compute_scores(y_true, y_pred, pr2_score)

    # Save plots and results
    output_dir = Path("parameter_decoding") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_plots:
        save_plots(r2_scores, r2_pr2_scores, model_name, layer, args.ridge_alpha, output_dir)

    data_out_name = (
        output_dir
        / f"{layer}_r2_decoding_values"
        f"{f'_ridge_alpha_{args.ridge_alpha:.0e}' if args.ridge_alpha != 0.0 else ''}.pkl"
    )
    data = {"r2_scores": r2_scores, "pr2_scores": r2_pr2_scores}

    with open(data_out_name, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        "--save_plots",
        action="store_true",
        help="Generate and save plots (disabled by default)",
    )
    args = parser.parse_args()
    main(args)
