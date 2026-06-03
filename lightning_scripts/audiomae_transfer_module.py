"""
Lightning module for training linear probes on top of a frozen AudioMAE encoder.

Mirrors WhisperTransferModule but uses the pretrained AudioMAE (hance-ai/audiomae)
ViT-Base encoder. Hooks at a specified ViT block, time-pools the 2D patch
activations, and trains linear (or MLP) classification heads for JSIN
word / speaker tasks.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from default_paths import JSIN_PATH, require_path
import torch.nn as nn
import lightning as L
import torchaudio

from audiomae_encoder_utils import (
    AUDIOMAE_DIM,
    AUDIOMAE_FREQ_PATCHES,
    AUDIOMAE_SR,
    AUDIOMAE_TIME_PATCHES,
    preprocess_waveform,
)
from jsinV3DataLoader_precombined_batched import jsinV3_precombined_all_signals
from torchmetrics.classification import Accuracy, BinaryPrecision
from robustness.audio_functions.jsinV3_loss_functions import jsinV3_multi_task_loss
import robustness.audio_functions.audio_transforms as at
from optimizers import LARS, CosineWarmupScheduler


class AudioMAETransferModule(L.LightningModule):
    def __init__(self, config, ckpt_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        from transformers import AutoModel
        from transformers.modeling_utils import PreTrainedModel

        _orig_mark = PreTrainedModel.mark_tied_weights_as_initialized
        def _safe_mark(self_model, loading_info):
            if not hasattr(self_model, "all_tied_weights_keys"):
                self_model.all_tied_weights_keys = {}
            return _orig_mark(self_model, loading_info)
        PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark

        wrapper = AutoModel.from_pretrained(
            "hance-ai/audiomae", trust_remote_code=True
        )
        self.audiomae_encoder = wrapper.encoder  # AudioMAEEncoder (VisionTransformer)

        for param in self.audiomae_encoder.parameters():
            param.requires_grad = False
        self.audiomae_encoder.eval()

        self.encoder_layer_idx = config["model"]["arch_kwargs"].get("encoder_layer", 11)
        self.n_blocks = len(self.audiomae_encoder.blocks)
        self.use_norm = self.encoder_layer_idx >= self.n_blocks

        self.encoder_activations = {}
        if not self.use_norm:
            self._register_encoder_hook()

        time_pool = config["model"]["arch_kwargs"].get("time_average", True)
        self.time_pool = time_pool
        if time_pool:
            self.classifier_input_dim = AUDIOMAE_DIM * AUDIOMAE_FREQ_PATCHES  # 6144
        else:
            self.classifier_input_dim = (
                AUDIOMAE_DIM * AUDIOMAE_FREQ_PATCHES * AUDIOMAE_TIME_PATCHES
            )  # 393216

        self.crop_audio = config.get("crop_audio", False)
        if self.crop_audio:
            self.audio_crop = at.CenterCropForegroundBackground(
                signal_size=40_000, crop_length=20_000
            )

        # JSIN data is at 20 kHz; AudioMAE expects 16 kHz
        self.resample_audio = lambda x: torchaudio.functional.resample(
            x, orig_freq=20_000, new_freq=AUDIOMAE_SR
        )

        self.transforms = at.AudioCompose([
            at.AudioToTensor(),
            at.CombineWithRandomDBSNR(low_snr=-10, high_snr=10),
            at.DBSPLNormalizeForegroundAndBackground(dbspl=60),
            at.UnsqueezeAudio(dim=0),
        ])

        self.test_transforms = at.AudioCompose([
            at.AudioToTensor(),
            at.CenterCropForegroundBackground(signal_size=40_000, crop_length=40_000),
            at.CombineWithRandomDBSNR(low_snr=-10, high_snr=10),
            at.DBSPLNormalizeForegroundAndBackground(dbspl=60),
            at.UnsqueezeAudio(dim=0),
        ])

        num_classes = config["model"]["arch_kwargs"]["num_classes"]
        proj_out_dim = self.classifier_input_dim

        self.mlp = None
        if config["model"].get("classifier", False):
            hidden_dims = [proj_out_dim] + config["model"]["classifier"]["hidden_dims"]
            layers = []
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                layers.append(nn.ReLU())
            proj_out_dim = hidden_dims[-1]
            self.mlp = nn.Sequential(*layers)

        if isinstance(num_classes, dict):
            all_fc_layers = {}
            for task in num_classes:
                all_fc_layers[task] = nn.Linear(proj_out_dim, num_classes[task])
            self.classifier = nn.ModuleDict(all_fc_layers)
        else:
            self.classifier = nn.Linear(proj_out_dim, num_classes)

        if config["model"].get("with_dropout", False):
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = False

        self.multi_task_loss = jsinV3_multi_task_loss(
            task_loss_params=config["hparas"]["task_loss_params"],
            batch_size=None,
        )

        self.accuracy = torch.nn.ModuleDict({
            task_key: BinaryPrecision()
            if "noise" in task_key
            else Accuracy(task="multiclass", num_classes=nc)
            for task_key, nc in self.config["model"]["arch_kwargs"]["num_classes"].items()
        })

    def _register_encoder_hook(self):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.encoder_activations["layer_output"] = out.detach()

        if self.encoder_layer_idx < 0 or self.encoder_layer_idx >= self.n_blocks:
            raise ValueError(
                f"Encoder layer index {self.encoder_layer_idx} out of range. "
                f"Model has {self.n_blocks} blocks (0-{self.n_blocks - 1})"
            )
        self.audiomae_encoder.blocks[self.encoder_layer_idx].register_forward_hook(hook_fn)

    def _extract_features(self, mel: torch.Tensor) -> torch.Tensor:
        """Run frozen AudioMAE and return flattened features from the hooked layer."""
        with torch.no_grad():
            self.encoder_activations.clear()
            full_out = self.audiomae_encoder.forward_features(mel)

            if self.use_norm:
                tokens = full_out
            else:
                tokens = self.encoder_activations["layer_output"]

            tokens = tokens[:, 1:, :]  # remove CLS -> (B, 512, 768)

            feats = tokens.reshape(
                tokens.shape[0],
                AUDIOMAE_FREQ_PATCHES,
                AUDIOMAE_TIME_PATCHES,
                AUDIOMAE_DIM,
            ).permute(0, 3, 1, 2)  # (B, D, freq, time)

            if self.time_pool:
                feats = feats.mean(dim=-1)  # (B, D, freq)

            activations = feats.flatten(start_dim=1).detach()
            self.encoder_activations.clear()

        return activations

    def forward(self, mel):
        activations = self._extract_features(mel)

        if self.dropout:
            activations = self.dropout(activations)
        if self.mlp:
            activations = self.mlp(activations)
        if isinstance(self.classifier, nn.ModuleDict):
            logits = {}
            for task, fc_l in self.classifier.items():
                logits[task] = fc_l(activations)
        else:
            logits = self.classifier(activations)
        return logits

    def _step(self, batch, batch_idx, step_type):
        mel, labels = batch
        logits = self.forward(mel)
        loss, task_loss_dict = self.multi_task_loss(logits, labels, return_indiv_loss=True)
        self.log(f"{step_type}_loss", loss.detach(), prog_bar=True, sync_dist=True)

        for task, task_loss in task_loss_dict.items():
            task_acc = self.accuracy[task](logits[task], labels[task])
            self.log(f"{step_type}_{task}_loss", task_loss.detach(), prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_{task}_acc", task_acc, prog_bar=False, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def predict_step(self, batch):
        mel, labels = batch
        logits = self.forward(mel)
        loss, task_loss_dict = self.multi_task_loss(logits, labels, return_indiv_loss=True)
        accuracy_dict = {}
        top5_dict = {}
        for task, task_loss in task_loss_dict.items():
            task_IXS = (labels[task] != 0).nonzero(as_tuple=True)
            task_logits = logits[task][task_IXS]
            task_labels = labels[task][task_IXS]
            task_acc = self.accuracy[task](task_logits, task_labels)
            accuracy_dict[task] = task_acc
            task_top5 = (
                torch.isin(
                    torch.topk(task_logits.softmax(-1), k=5, dim=-1).indices,
                    task_labels,
                )
                .any(-1)
                .float()
                .mean()
            )
            top5_dict[task] = task_top5
        return {"top1": accuracy_dict, "top5": top5_dict}

    def configure_optimizers(self):
        if self.config["hparas"]["optimizer"] == "LARS":
            lr = self.config["hparas"]["lr"] * self.config["hparas"]["global_batch_size"] / 256
            self.optimizer = LARS(
                self.classifier.parameters(),
                lr=lr,
                weight_decay=1e-6,
                momentum=0.9,
                weight_decay_filter=True,
                lars_adaptation_filter=True,
            )
        else:
            lr = self.config["hparas"]["lr"]
            opt = getattr(torch.optim, self.config["hparas"]["optimizer"])
            self.optimizer = opt(self.classifier.parameters(), lr=lr)

        if self.config["hparas"].get("lr_schedule", False):
            total_training_steps = self.total_training_steps()
            num_warmup_steps = self.compute_warmup(
                total_training_steps, self.config["hparas"]["num_warmup_steps_or_ratio"]
            )
            lr_scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                batch_size=self.config["hparas"]["global_batch_size"],
                warmup_steps=num_warmup_steps,
                max_steps=total_training_steps,
                lr=lr,
            )
            return [self.optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        return [self.optimizer]

    def _waveforms_to_mel(self, signals: torch.Tensor) -> torch.Tensor:
        """Convert a batch of waveforms (already at 16 kHz) to AudioMAE mel input."""
        return preprocess_waveform(signals, sr=AUDIOMAE_SR)

    def collate_fn(self, batch):
        batch = batch[0]
        signals = []
        labels = batch[-1]

        if isinstance(labels, dict):
            for task_key, task_labels in labels.items():
                labels[task_key] = torch.from_numpy(task_labels)
        else:
            labels = torch.from_numpy(labels)

        for signal, noise in zip(*batch[:2]):
            if not self.config.get("with_noise", False):
                noise = None
            if self.crop_audio:
                signal, noise = self.audio_crop(signal, noise)
                signal = at.pad_or_trim_to_len(signal, 40000, mode="both")
                if noise is not None:
                    noise = at.pad_or_trim_to_len(noise, 40000, mode="both")
            signal, _ = self.transforms(signal, noise)
            if signal is None:
                signal = torch.zeros(1, 40000)
            signal = self.resample_audio(signal.unsqueeze(0)).squeeze(0)
            signals.append(signal)

        signals = torch.cat(signals, dim=0)  # (B, T)
        mel = self._waveforms_to_mel(signals)  # (B, 1, 1024, 128)
        return mel, labels

    def eval_collate_fn(self, batch):
        audio, targets = batch[0]
        labels = {}
        if isinstance(targets, dict):
            for task_key, task_labels in targets.items():
                labels[task_key] = torch.from_numpy(task_labels)
        else:
            labels = torch.from_numpy(targets)

        audio = self.resample_audio(audio)
        mel = self._waveforms_to_mel(audio)  # (B, 1, 1024, 128)
        return mel, labels

    def train_dataloader(self):
        dataset = jsinV3_precombined_all_signals(
            root=str(require_path(JSIN_PATH, "COCHDNN_JSIN_DIR", "JSIN/WSN dataset")),
            train=True,
            transform=None,
            batch_size=self.config["hparas"]["batch_size"],
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        dataset = jsinV3_precombined_all_signals(
            root=str(require_path(JSIN_PATH, "COCHDNN_JSIN_DIR", "JSIN/WSN dataset")),
            train=False,
            transform=None,
            batch_size=self.config["hparas"]["batch_size"],
            eval_max=self.config["data"].get("eval_max", 3),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def total_training_steps(self) -> int:
        dataset_size = len(self.train_dataloader())
        num_devices = self.config["num_gpus"]
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps and self.trainer.max_steps != -1:
            return int(self.trainer.max_steps)
        return int(max_estimated_steps)

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        return (
            num_warmup_steps * num_training_steps
            if isinstance(num_warmup_steps, float)
            else num_warmup_steps
        )
