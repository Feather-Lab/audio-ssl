"""
Lightning modules for training linear probes on frozen AudioMAE or Whisper
features for Speech Commands 30-class word recognition.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import lightning as L
from datasets import load_dataset
from torchmetrics.classification import Accuracy

from audiomae_encoder_utils import (
    AUDIOMAE_DIM,
    AUDIOMAE_FREQ_PATCHES,
    AUDIOMAE_SR,
    AUDIOMAE_TIME_PATCHES,
    preprocess_waveform,
)
from optimizers import LARS, CosineWarmupScheduler
import whisper
from whisper_encoder_arch import WHISPER_SR

NUM_CLASSES = 30  # speech_commands v0.01


class CenterCropOrPad:
    def __init__(self, sig_length):
        self.sig_length = sig_length

    def __call__(self, x):
        if x.shape[0] < self.sig_length:
            pad_dur = (self.sig_length - len(x)) // 2 + 1
            x = nn.functional.pad(x, (pad_dur, pad_dur), "constant", 0)
        start_idx = int((x.shape[0] - self.sig_length) / 2)
        x = x[start_idx : start_idx + self.sig_length]
        return x


class AudioMAESpeechCommandsClassifier(L.LightningModule):
    """Train a linear classifier on frozen AudioMAE features for Speech Commands."""

    def __init__(self, config):
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
        self.audiomae_encoder = wrapper.encoder
        for param in self.audiomae_encoder.parameters():
            param.requires_grad = False
        self.audiomae_encoder.eval()

        self.encoder_layer_idx = config["encoder_layer"]
        self.n_blocks = len(self.audiomae_encoder.blocks)
        self.use_norm = self.encoder_layer_idx >= self.n_blocks

        self.encoder_activations = {}
        if not self.use_norm:
            self._register_hook()

        time_pool = config.get("time_average", True)
        self.time_pool = time_pool
        if time_pool:
            feat_dim = AUDIOMAE_DIM * AUDIOMAE_FREQ_PATCHES  # 6144
        else:
            feat_dim = AUDIOMAE_DIM * AUDIOMAE_FREQ_PATCHES * AUDIOMAE_TIME_PATCHES

        self.speech_commands_dataset = load_dataset(
            "google/speech_commands", "v0.01", trust_remote_code=True
        )
        self.crop_or_pad = CenterCropOrPad(16_000)  # 1 s at 16 kHz

        self.mlp = None
        proj_out_dim = feat_dim
        if config.get("classifier_hidden_dims"):
            hidden_dims = [feat_dim] + config["classifier_hidden_dims"]
            layers = []
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                layers.append(nn.ReLU())
            proj_out_dim = hidden_dims[-1]
            self.mlp = nn.Sequential(*layers)

        self.classifier = nn.Linear(proj_out_dim, NUM_CLASSES)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)

    def _register_hook(self):
        def hook_fn(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.encoder_activations["layer_output"] = out.detach()

        if self.encoder_layer_idx < 0 or self.encoder_layer_idx >= self.n_blocks:
            raise ValueError(
                f"encoder_layer {self.encoder_layer_idx} out of range "
                f"(model has {self.n_blocks} blocks, 0-{self.n_blocks - 1})"
            )
        self.audiomae_encoder.blocks[self.encoder_layer_idx].register_forward_hook(hook_fn)

    def _extract_features(self, mel: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.encoder_activations.clear()
            full_out = self.audiomae_encoder.forward_features(mel)

            if self.use_norm:
                tokens = full_out
            else:
                tokens = self.encoder_activations["layer_output"]

            tokens = tokens[:, 1:, :]  # remove CLS
            feats = tokens.reshape(
                tokens.shape[0],
                AUDIOMAE_FREQ_PATCHES,
                AUDIOMAE_TIME_PATCHES,
                AUDIOMAE_DIM,
            ).permute(0, 3, 1, 2)  # (B, D, freq, time)

            if self.time_pool:
                feats = feats.mean(dim=-1)

            activations = feats.flatten(start_dim=1).detach()
            self.encoder_activations.clear()
        return activations

    def forward(self, mel):
        activations = self._extract_features(mel)
        if self.mlp:
            activations = self.mlp(activations)
        return self.classifier(activations)

    def _step(self, batch, batch_idx, step_type):
        mel, labels = batch
        logits = self.forward(mel)
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log(f"{step_type}_loss", loss.detach(), prog_bar=True, sync_dist=True)
        self.log(f"{step_type}_acc", acc, prog_bar=True, sync_dist=True)
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
        task_acc = self.accuracy(logits, labels)
        task_top5 = (
            torch.isin(
                torch.topk(logits.softmax(-1), k=5, dim=-1).indices, labels
            )
            .any(-1)
            .float()
            .mean()
        )
        return {"top1": task_acc, "top5": task_top5}

    def configure_optimizers(self):
        if self.config["optimizer"] == "LARS":
            lr = self.config["lr"] * self.config["batch_size"] / 256
            self.optimizer = LARS(
                self.classifier.parameters(),
                lr=lr,
                weight_decay=1e-6,
                momentum=0.9,
                weight_decay_filter=True,
                lars_adaptation_filter=True,
            )
        else:
            lr = self.config["lr"]
            opt = getattr(torch.optim, self.config["optimizer"])
            self.optimizer = opt(self.classifier.parameters(), lr=lr)

        if self.config.get("lr_schedule", False):
            total_steps = self.total_training_steps()
            warmup = self.compute_warmup(total_steps, 0)
            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                batch_size=self.config["batch_size"],
                warmup_steps=warmup,
                max_steps=total_steps,
                lr=lr,
            )
            return [self.optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return [self.optimizer]

    def collate_fn(self, batch):
        mels, labels = [], []
        for eg in batch:
            wav = torch.from_numpy(eg["audio"]["array"]).float()
            wav = self.crop_or_pad(wav)
            mels.append(wav.unsqueeze(0))
            labels.append(eg["label"])
        waveforms = torch.stack(mels)
        mel = preprocess_waveform(waveforms.squeeze(1), sr=AUDIOMAE_SR)
        return mel, torch.tensor(labels)

    def _filter_silence(self, split):
        return split.filter(
            lambda eg: ("_silence_" not in eg["file"]) and (eg["audio"]["array"] is not None)
        )

    def train_dataloader(self):
        split = self._filter_silence(self.speech_commands_dataset["train"])
        return torch.utils.data.DataLoader(
            split,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        split = self._filter_silence(self.speech_commands_dataset["validation"])
        return torch.utils.data.DataLoader(
            split,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        split = self._filter_silence(self.speech_commands_dataset["test"])
        return torch.utils.data.DataLoader(
            split,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def total_training_steps(self) -> int:
        dataset_size = len(self.train_dataloader())
        num_devices = max(self.config.get("num_gpus", 1), 1)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_steps:
            return int(self.trainer.max_steps)
        return int(max_steps)

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        if isinstance(num_warmup_steps, float):
            return int(num_warmup_steps * num_training_steps)
        return num_warmup_steps


class WhisperSpeechCommandsClassifier(L.LightningModule):
    """Train a linear classifier on frozen pretrained Whisper features for Speech Commands."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        whisper_model_name = config.get("whisper_model", "large-v3")
        model = whisper.load_model(whisper_model_name)
        self.whisper_encoder = model.encoder
        self.n_mels = model.dims.n_mels
        n_audio_state = model.dims.n_audio_state
        del model

        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        self.whisper_encoder.eval()

        self.encoder_layer_idx = config["encoder_layer"]
        self.n_blocks = len(self.whisper_encoder.blocks)
        self.use_ln_post = self.encoder_layer_idx >= self.n_blocks

        self.encoder_activations = {}
        if not self.use_ln_post:
            self._register_hook()

        self.n_time_tokens = 100  # 2 seconds at 50 Hz
        feat_dim = self.n_time_tokens * n_audio_state

        self.speech_commands_dataset = load_dataset(
            "google/speech_commands", "v0.01", trust_remote_code=True
        )
        self.crop_or_pad = CenterCropOrPad(16_000)  # 1s at 16 kHz

        self.mlp = None
        proj_out_dim = feat_dim
        if config.get("classifier_hidden_dims"):
            hidden_dims = [feat_dim] + config["classifier_hidden_dims"]
            layers = []
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                layers.append(nn.ReLU())
            proj_out_dim = hidden_dims[-1]
            self.mlp = nn.Sequential(*layers)

        self.classifier = nn.Linear(proj_out_dim, NUM_CLASSES)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)

    def _register_hook(self):
        def hook_fn(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.encoder_activations["layer_output"] = out.detach()

        self.whisper_encoder.blocks[self.encoder_layer_idx].register_forward_hook(hook_fn)

    def _extract_features(self, mel: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.encoder_activations.clear()
            encoder_out = self.whisper_encoder(mel)

            if self.use_ln_post:
                act = encoder_out
            else:
                act = self.encoder_activations["layer_output"]

            act = act[:, : self.n_time_tokens, :]
            activations = act.flatten(start_dim=1).detach()
            self.encoder_activations.clear()
        return activations

    def forward(self, mel):
        activations = self._extract_features(mel)
        if self.mlp:
            activations = self.mlp(activations)
        return self.classifier(activations)

    def _step(self, batch, batch_idx, step_type):
        mel, labels = batch
        logits = self.forward(mel)
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log(f"{step_type}_loss", loss.detach(), prog_bar=True, sync_dist=True)
        self.log(f"{step_type}_acc", acc, prog_bar=True, sync_dist=True)
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
        task_acc = self.accuracy(logits, labels)
        task_top5 = (
            torch.isin(
                torch.topk(logits.softmax(-1), k=5, dim=-1).indices, labels
            )
            .any(-1)
            .float()
            .mean()
        )
        return {"top1": task_acc, "top5": task_top5}

    def configure_optimizers(self):
        if self.config["optimizer"] == "LARS":
            lr = self.config["lr"] * self.config["batch_size"] / 256
            self.optimizer = LARS(
                self.classifier.parameters(),
                lr=lr,
                weight_decay=1e-6,
                momentum=0.9,
                weight_decay_filter=True,
                lars_adaptation_filter=True,
            )
        else:
            lr = self.config["lr"]
            opt = getattr(torch.optim, self.config["optimizer"])
            self.optimizer = opt(self.classifier.parameters(), lr=lr)

        if self.config.get("lr_schedule", False):
            total_steps = self.total_training_steps()
            warmup = self.compute_warmup(total_steps, 0)
            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                batch_size=self.config["batch_size"],
                warmup_steps=warmup,
                max_steps=total_steps,
                lr=lr,
            )
            return [self.optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return [self.optimizer]

    def collate_fn(self, batch):
        mels, labels = [], []
        for eg in batch:
            wav = torch.from_numpy(eg["audio"]["array"]).float()
            wav = self.crop_or_pad(wav)
            mels.append(wav.unsqueeze(0))
            labels.append(eg["label"])
        waveforms = torch.cat(mels, dim=0)  # (B, T)
        waveforms = whisper.pad_or_trim(waveforms)
        mel = whisper.log_mel_spectrogram(waveforms, n_mels=self.n_mels)
        return mel, torch.tensor(labels)

    def _filter_silence(self, split):
        return split.filter(
            lambda eg: ("_silence_" not in eg["file"]) and (eg["audio"]["array"] is not None)
        )

    def train_dataloader(self):
        split = self._filter_silence(self.speech_commands_dataset["train"])
        return torch.utils.data.DataLoader(
            split,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        split = self._filter_silence(self.speech_commands_dataset["validation"])
        return torch.utils.data.DataLoader(
            split,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        split = self._filter_silence(self.speech_commands_dataset["test"])
        return torch.utils.data.DataLoader(
            split,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def total_training_steps(self) -> int:
        dataset_size = len(self.train_dataloader())
        num_devices = max(self.config.get("num_gpus", 1), 1)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_steps:
            return int(self.trainer.max_steps)
        return int(max_steps)

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        if isinstance(num_warmup_steps, float):
            return int(num_warmup_steps * num_training_steps)
        return num_warmup_steps
