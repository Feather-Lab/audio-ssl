"""
ESC-50 SVM linear probe evaluation for pretrained AudioMAE or Whisper encoders.

Mirrors make_esc_pl_model_plots.py but uses AudioMAE (hance-ai/audiomae)
or pretrained Whisper with layerwise activation extraction.  For each
specified layer, extracts time-pooled features for every ESC-50 clip,
runs 5-fold cross-validated LinearSVC, and saves results + confusion
matrix plots.

Usage (AudioMAE):
    python eval_audiomae_esc50.py -L 12 -D ${COCHDNN_SCRATCH_DIR:-/tmp/cochdnn} \
        -A 4096 -R 5 -P -C 0.01 0.1 1 10 100

Usage (Whisper large-v3):
    python eval_audiomae_esc50.py --model_type whisper --whisper_model large-v3 \
        --layer_name ln_post -D ${COCHDNN_SCRATCH_DIR:-/tmp/cochdnn} -A 4096 -R 5 -P -C 0.01 0.1 1 10 100
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from sklearn import metrics, preprocessing, svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier

from default_paths import ESC50_DIR, require_path

matplotlib.rcParams.update({"font.size": 26})
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lightning_scripts"))

from lightning_scripts.audiomae_encoder_utils import (
    AUDIOMAE_DIM,
    AUDIOMAE_FREQ_PATCHES,
    AUDIOMAE_SR,
    AUDIOMAE_TIME_PATCHES,
    AudioMAELayerwiseEncoder,
    preprocess_waveform,
)
from lightning_scripts.whisper_encoder_arch import WhisperLayerwiseEncoder, WHISPER_SR
from robustness.tools.audio_helpers import load_audio_wav_resample

ESC50_ROOT = require_path(ESC50_DIR, "COCHDNN_ESC50_DIR", "ESC-50 dataset")
ESC50_DATA_PATH = ESC50_ROOT / "audio"
ESC50_FOLD_PATH = ESC50_ROOT / "meta" / "esc50.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_train_and_test(left_out_fold):
    df = pandas.read_csv(ESC50_FOLD_PATH)
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    train_ids, test_ids = [], []

    for _, row in df.iterrows():
        if row["fold"] == left_out_fold:
            test_paths.append(str(ESC50_DATA_PATH / row["filename"]))
            test_labels.append(row["target"])
            test_ids.append(row["filename"])
        else:
            train_paths.append(str(ESC50_DATA_PATH / row["filename"]))
            train_labels.append(row["target"])
            train_ids.append(row["filename"])

    return train_paths, train_labels, test_paths, test_labels, {"train": train_ids, "test": test_ids}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    encoder,
    audio_paths: list[str],
    layer: str,
    num_reps: int,
    seed: int,
    scratch_dir: str,
    sound_ids: list[str],
    overwrite: bool,
    dur_secs: float = 2.0,
    device: torch.device = torch.device("cpu"),
    model_type: str = "audiomae",
) -> np.ndarray:
    """Extract encoder features for a list of audio files.

    Returns:
        If num_reps == 1: array of shape (n_sounds, feature_dim)
        If num_reps > 1:  array of shape (n_sounds, num_reps, feature_dim)
    """
    np.random.seed(seed * 2)
    cache_subdir = f"{model_type}_activations"
    cache_dir = Path(scratch_dir) / cache_subdir / layer
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_sr = WHISPER_SR if model_type == "whisper" else AUDIOMAE_SR

    all_feats = []
    for idx, audio_path in enumerate(audio_paths):
        cache_file = cache_dir / f"reps{num_reps}_rs{seed}_{sound_ids[idx]}.npy"

        if cache_file.exists() and not overwrite:
            feats = np.load(cache_file)
        else:
            rep_feats = []
            for _ in range(num_reps):
                sound, sr = load_audio_wav_resample(
                    audio_path,
                    resample_SR=target_sr,
                    DUR_SECS=dur_secs,
                    START_SECS="random",
                )
                while sound.sum() == 0:
                    sound, sr = load_audio_wav_resample(
                        audio_path,
                        resample_SR=target_sr,
                        DUR_SECS=dur_secs,
                        START_SECS="random",
                    )
                sound = sound - np.mean(sound)
                rms = np.sqrt(np.mean(sound**2))
                if rms > 0:
                    sound = sound / rms * 0.1

                waveform = torch.from_numpy(sound).float().unsqueeze(0)  # (1, T)

                if model_type == "whisper":
                    embeddings = encoder(waveform.to(device), sr=target_sr, flatten_activations=False)
                    act = embeddings[layer]  # (1, n_ctx, n_state)
                    pooled = act.mean(dim=1).cpu().numpy().ravel()  # time-pool
                else:
                    mel = preprocess_waveform(waveform, sr=AUDIOMAE_SR).to(device)
                    encoder._block_outputs.clear()
                    with torch.no_grad():
                        full_out = encoder.encoder.forward_features(mel)

                    if layer == "norm":
                        tokens = full_out
                    else:
                        tokens = encoder._block_outputs[layer]

                    tokens = tokens[:, 1:, :]  # remove CLS
                    feats_4d = tokens.reshape(
                        1, AUDIOMAE_FREQ_PATCHES, AUDIOMAE_TIME_PATCHES, AUDIOMAE_DIM
                    ).permute(0, 3, 1, 2)
                    pooled = feats_4d.mean(dim=-1).cpu().numpy().ravel()
                    encoder._block_outputs.clear()

                rep_feats.append(pooled)

            feats = np.array(rep_feats)  # (num_reps, feature_dim)
            np.save(cache_file, feats)

        all_feats.append(feats)

    all_feats = np.array(all_feats)  # (n_sounds, num_reps, feature_dim) or (n_sounds, 1, feat)
    if num_reps == 1:
        all_feats = all_feats.squeeze(1)
    return all_feats


# ---------------------------------------------------------------------------
# SVM training (reused from make_esc_pl_model_plots.py)
# ---------------------------------------------------------------------------

def _avg_scorer_with_nrep(n_rep, estimator, X, y):
    y_prob = estimator.decision_function(X)
    y_prob = y_prob.reshape(int(y_prob.shape[0] / n_rep), n_rep, -1)
    y_prob = np.mean(y_prob, axis=1)
    y_c = np.argmax(y, axis=1).reshape(-1, n_rep)[:, 0]
    pred = np.argmax(y_prob, axis=1)
    return metrics.accuracy_score(y_c, pred)


def _avg_scorer(estimator, X, y):
    y_prob = estimator.decision_function(X)
    y_c = np.argmax(y, axis=1)
    pred = np.argmax(y_prob, axis=1)
    return metrics.accuracy_score(y_c, pred)


def train_svm(train_features, train_labels, test_features, test_labels,
              c_values, average_test_predictions, n_splits_cv=3):
    train_shape = train_features.shape
    test_shape = test_features.shape

    if len(train_shape) > 2:
        n_rep = train_shape[1]
        train_features = train_features.reshape(train_shape[0] * n_rep, train_shape[2])
        train_labels = np.repeat(train_labels, n_rep)

        cv_sounds = ShuffleSplit(n_splits=n_splits_cv, test_size=0.25, random_state=0)
        cv_index = cv_sounds.split(np.arange(train_shape[0]))
        cv_list = []
        for v in cv_index:
            cv_list.append([
                np.array([n_rep * k + np.arange(n_rep) for k in v[0]]).ravel(),
                np.array([n_rep * k + np.arange(n_rep) for k in v[1]]).ravel(),
            ])
        cv = iter(cv_list)

        test_features = test_features.reshape(test_shape[0] * n_rep, test_shape[2])
        test_labels = np.repeat(test_labels, n_rep)

        scorer = partial(_avg_scorer_with_nrep, n_rep) if average_test_predictions else _avg_scorer
    else:
        cv = ShuffleSplit(n_splits=n_splits_cv, test_size=0.25, random_state=0)
        scorer = _avg_scorer

    lb = preprocessing.LabelBinarizer()
    lb.fit(range(50))
    test_labels_bin = lb.transform(test_labels)
    train_labels_bin = lb.transform(train_labels)

    svc = OneVsRestClassifier(
        svm.LinearSVC(max_iter=1000, dual=True, random_state=0), n_jobs=4
    )
    clf = GridSearchCV(
        svc, param_grid={"estimator__C": c_values}, cv=cv, n_jobs=4, scoring=scorer, refit=True
    )
    tic = time.perf_counter()
    clf.fit(train_features, train_labels_bin)
    print(f"SVM fit in {time.perf_counter() - tic:.1f}s")

    predictions = lb.transform(np.argmax(clf.decision_function(test_features), 1))
    if len(test_shape) > 2 and average_test_predictions:
        predictions = np.mean(
            predictions.reshape(test_shape[0], test_shape[1], -1), axis=1
        )
    predictions = np.argmax(predictions, axis=1)
    acc = clf.score(test_features, test_labels_bin)
    return predictions, acc


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_esc50_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_type = getattr(args, "model_type", "audiomae")

    if model_type == "whisper":
        whisper_model = getattr(args, "whisper_model", "large-v3")
        print(f"Loading Whisper ({whisper_model}) …")
        encoder = WhisperLayerwiseEncoder(whisper_model_name=whisper_model).to(device)
        net_name = f"whisper_{whisper_model}"
    else:
        print("Loading AudioMAE …")
        encoder = AudioMAELayerwiseEncoder(time_pool=True).to(device)
        net_name = "audiomae"

    layer_names = encoder.layer_names
    if args.layer_idx is not None:
        layer = layer_names[args.layer_idx]
    else:
        layer = args.layer_name
    print(f"Evaluating layer: {layer}")
    folds = [1, 2, 3, 4, 5]

    all_predictions, all_labels, avg_accuracies = [], [], []

    for fold in folds:
        print(f"\n--- Fold {fold} ---")
        train_paths, train_labels, test_paths, test_labels, sound_ids = get_train_and_test(fold)

        overwrite = args.overwrite if fold == 1 else False

        train_feats = extract_features(
            encoder, train_paths, layer, args.num_reps, args.seed,
            args.scratch_dir, sound_ids["train"], overwrite,
            dur_secs=args.dur_secs, device=device, model_type=model_type,
        )
        test_feats = extract_features(
            encoder, test_paths, layer, args.num_reps, args.seed,
            args.scratch_dir, sound_ids["test"], overwrite,
            dur_secs=args.dur_secs, device=device, model_type=model_type,
        )

        # Normalize
        if len(train_feats.shape) > 2:
            orig = train_feats.shape
            scaler = preprocessing.StandardScaler().fit(
                train_feats.reshape(orig[0] * orig[1], orig[2])
            )
            train_feats = scaler.transform(
                train_feats.reshape(orig[0] * orig[1], orig[2])
            ).reshape(orig)
            torig = test_feats.shape
            test_feats = scaler.transform(
                test_feats.reshape(torig[0] * torig[1], torig[2])
            ).reshape(torig)
        else:
            scaler = preprocessing.StandardScaler().fit(train_feats)
            train_feats = scaler.transform(train_feats)
            test_feats = scaler.transform(test_feats)

        preds, acc = train_svm(
            train_feats, train_labels, test_feats, test_labels,
            args.c_values, args.avg_test_predictions,
        )
        all_predictions.append(preds)
        all_labels.append(test_labels)
        avg_accuracies.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")

    overall_acc = np.mean(avg_accuracies)
    print(f"\nOverall accuracy: {overall_acc:.4f}")

    # Target-to-category mapping
    df = pandas.read_csv(ESC50_FOLD_PATH)
    target_to_category = {}
    for _, row in df.iterrows():
        if row["target"] not in target_to_category:
            target_to_category[row["target"]] = row["category"]

    # Per-category accuracy
    category_accuracies = []
    for fold_idx in range(5):
        fold_accs = []
        for target in range(50):
            correct = sum(
                1 for p, l in zip(all_predictions[fold_idx], all_labels[fold_idx])
                if l == target and p == l
            )
            total = sum(1 for l in all_labels[fold_idx] if l == target)
            fold_accs.append(correct / total if total > 0 else 0)
        category_accuracies.append(fold_accs)

    # Save results
    save_path = Path(f"esc_analysis/{net_name}_{layer}_nact{args.num_activations}_"
                     f"nreps{args.num_reps}_rs{args.seed}_avgtest{args.avg_test_predictions}")
    save_path.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "overall_acc": overall_acc,
        "avg_accuracies": avg_accuracies,
        "category_accuracies": category_accuracies,
        "target_to_category": target_to_category,
        "all_predictions": all_predictions,
        "all_test_labels": all_labels,
    }
    with open(save_path / f"saved_vars_{layer}.pickle", "wb") as f:
        pickle.dump(result_dict, f)

    # Plot per-category accuracy
    category_arr = np.array(category_accuracies)
    category_list = np.array(list(target_to_category.values()))
    means = np.mean(category_arr, axis=0)
    sems = np.std(category_arr, axis=0)

    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    ax.bar(category_list, means, yerr=sems, align="center", alpha=0.5, ecolor="black", capsize=10)
    ax.set_title(f"Average Per Category Accuracy: Layer {layer}")
    ax.axhline(overall_acc, color="red", linewidth=2)
    plt.xticks(category_list, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path / f"avg_per_category_accuracy_{layer}.pdf", bbox_inches="tight", transparent=True)
    plt.close()

    # Confusion matrix
    target_list = np.array(list(target_to_category.keys()))
    conf_matrices = []
    for fold_idx in range(5):
        c = confusion_matrix(all_labels[fold_idx], all_predictions[fold_idx],
                             labels=target_list, normalize="true")
        conf_matrices.append(c)
    avg_conf = np.mean(conf_matrices, axis=0)

    fig, ax = plt.subplots(figsize=(25, 25))
    plt.imshow(avg_conf, cmap="Blues")
    plt.colorbar()
    ax.set_xticks(np.arange(50))
    ax.set_yticks(np.arange(50))
    ax.set_xticklabels(category_list)
    ax.set_yticklabels(category_list)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    ax.set_title(f"Confusion matrix for layer {layer}")
    fig.tight_layout()
    plt.savefig(save_path / f"confusion_matrix_{layer}.pdf", bbox_inches="tight", transparent=True)
    plt.close()

    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESC-50 SVM evaluation (AudioMAE or Whisper)")
    parser.add_argument("--model_type", type=str, default="audiomae",
                        choices=["audiomae", "whisper"],
                        help="Encoder to evaluate: audiomae or whisper.")
    parser.add_argument("--whisper_model", type=str, default="large-v3",
                        help="Whisper model name (e.g. large-v3). Only used when model_type=whisper.")
    parser.add_argument("-L", "--layer_idx", type=int, default=None,
                        help="Layer index. AudioMAE: 0-11=blocks, 12=norm. "
                             "Whisper: 0..N-1=encoder blocks, N=ln_post. Overrides --layer_name.")
    parser.add_argument("--layer_name", type=str, default="block_11",
                        help="Layer name (e.g. block_5, norm, ln_post). Used if --layer_idx not given.")
    parser.add_argument("-D", "--scratch_dir", type=str, required=True,
                        help="Scratch directory for cached activations.")
    parser.add_argument("-A", "--num_activations", type=int, default=4096,
                        help="Not used for downsampling (AudioMAE features are 6144-dim), "
                             "kept for naming compatibility.")
    parser.add_argument("-R", "--num_reps", type=int, default=5,
                        help="Number of random 2s clips per sound.")
    parser.add_argument("-S", "--seed", type=int, default=0)
    parser.add_argument("-P", "--avg_test_predictions", action="store_true")
    parser.add_argument("-C", "--c_values", type=float, nargs="+",
                        default=[0.01, 0.1, 1, 10, 100])
    parser.add_argument("-O", "--overwrite", action="store_true")
    parser.add_argument("--dur_secs", type=float, default=2.0,
                        help="Duration of audio clips in seconds.")
    args = parser.parse_args()
    run_esc50_eval(args)
