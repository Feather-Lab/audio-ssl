"""
NSynth linear probe for the pretrained AudioMAE encoder.

Mirrors eval_nsynth_linear.py but uses AudioMAE (hance-ai/audiomae) with
activation extraction from a specified ViT block. Trains a linear classifier
on frozen, time-pooled features for instrument family (11 classes) or other
NSynth classification tasks.
"""

from __future__ import annotations

import os
import pickle
import pathlib
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import Accuracy

from audiomae_encoder_utils import (
    AUDIOMAE_DIM,
    AUDIOMAE_FREQ_PATCHES,
    AUDIOMAE_SR,
    AUDIOMAE_TIME_PATCHES,
    preprocess_waveform,
)
from nsynth_dataset import NsynthDataset
from audio_ssl.misc import LARS, CosineWarmupScheduler

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class AudioMAENSynthClassifier(L.LightningModule):
    """Train a linear classifier on frozen AudioMAE features for NSynth tasks."""

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
        self.use_norm = self.encoder_layer_idx == self.n_blocks

        self.encoder_activations = {}
        if not self.use_norm:
            self._register_hook()

        time_pool = config.get("time_average", True)
        self.time_pool = time_pool
        if time_pool:
            feat_dim = AUDIOMAE_DIM * AUDIOMAE_FREQ_PATCHES  # 6144
        else:
            feat_dim = AUDIOMAE_DIM * AUDIOMAE_FREQ_PATCHES * AUDIOMAE_TIME_PATCHES

        num_classes = config["num_classes"]

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

        if config.get("with_dropout", False):
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.classifier = nn.Linear(proj_out_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

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
            ).permute(0, 3, 1, 2)

            if self.time_pool:
                feats = feats.mean(dim=-1)

            activations = feats.flatten(start_dim=1).detach()
            self.encoder_activations.clear()
        return activations

    def forward(self, mel):
        activations = self._extract_features(mel)
        if self.dropout is not None:
            activations = self.dropout(activations)
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
        k = min(5, self.config["num_classes"])
        task_top5 = (
            torch.isin(
                torch.topk(logits.softmax(-1), k=k, dim=-1).indices, labels
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

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _make_dataset(self, split: str) -> NsynthDataset:
        return NsynthDataset(
            nsynth_root=self.config["nsynth_root"],
            split=split,
            task=self.config["task"],
            label_field=self.config.get("label_field"),
            sample_rate=AUDIOMAE_SR,
            duration=self.config.get("duration"),
            fade_window_duration=self.config.get("fade_window_duration", 0.01),
        )

    def _collate(self, batch):
        waveforms, labels = zip(*batch)
        waveforms = torch.stack(waveforms)  # (B, T)
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        mel = preprocess_waveform(waveforms, sr=AUDIOMAE_SR)
        return mel, torch.tensor(labels)

    def train_dataloader(self):
        ds = self._make_dataset("train")
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        ds = self._make_dataset("valid")
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            collate_fn=self._collate,
        )

    def test_dataloader(self):
        ds = self._make_dataset("test")
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            collate_fn=self._collate,
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


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def cli_main(args):
    L.seed_everything(args.random_seed)

    layer_idx = args.layer_idx
    layer_name = f"block_{layer_idx}" if layer_idx < 12 else "norm"

    train_dataset = NsynthDataset(
        nsynth_root=args.nsynth_root,
        split="train",
        task=args.task,
        label_field=args.label_field if args.task == "other" else None,
        sample_rate=AUDIOMAE_SR,
        duration=args.duration,
        fade_window_duration=args.fade_window_duration,
    )
    num_classes = train_dataset.num_classes
    print(f"NSynth task '{args.task}': {num_classes} classes")

    config = {
        "encoder_layer": layer_idx,
        "time_average": True,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "lr_schedule": args.lr_scheduler,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_gpus": args.gpus,
        "num_classes": num_classes,
        "nsynth_root": args.nsynth_root,
        "task": args.task,
        "label_field": args.label_field if args.task == "other" else None,
        "duration": args.duration,
        "fade_window_duration": args.fade_window_duration,
        "classifier_hidden_dims": [args.mlp_dim] if args.w_mlp else None,
        "with_dropout": args.with_dropout,
    }

    module = AudioMAENSynthClassifier(config=config)

    ckpt_dir = pathlib.Path(args.model_ckpt_dir) / f"audiomae/nsynth_{args.task}/{layer_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    existing_ckpts = sorted(ckpt_dir.rglob("*.ckpt"), key=os.path.getctime)
    if existing_ckpts and args.use_classifier_ckpt:
        ckpt = torch.load(str(existing_ckpts[-1]), weights_only=False)
        module.load_state_dict(ckpt["state_dict"])
        print(f"Loaded classifier from {existing_ckpts[-1]}")

    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

    callbacks = [
        ModelCheckpoint(
            ckpt_dir,
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    from pytorch_lightning.loggers import WandbLogger

    wandb_logger = WandbLogger(
        save_dir=str(ckpt_dir),
        name=f"audiomae_nsynth_{args.task}_{layer_name}",
        group="audiomae_nsynth",
        project="cochdnn",
    )

    trainer = L.Trainer(
        precision="32",
        default_root_dir=str(ckpt_dir),
        max_epochs=args.train_epochs,
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp" if args.gpus > 1 else "auto",
        gradient_clip_val=1,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    if not args.eval_only:
        trainer.fit(module)

    print("Running test inference")
    test_dl = module.test_dataloader()
    outputs = trainer.predict(module, test_dl, return_predictions=True)

    top1_scores = [r["top1"].item() for r in outputs]
    top5_scores = [r["top5"].item() for r in outputs]

    def bootstrap_sem(scores, n=1000):
        mean = np.mean(scores)
        boots = [np.mean(np.random.choice(scores, size=len(scores))) for _ in range(n)]
        return mean, np.std(boots)

    top1_mean, top1_sem = bootstrap_sem(top1_scores)
    top5_mean, top5_sem = bootstrap_sem(top5_scores)

    output_dict = {
        "top1_mean": top1_mean,
        "top1_sem": top1_sem,
        "top5_mean": top5_mean,
        "top5_sem": top5_sem,
        "num_classes": num_classes,
        "task": args.task,
        "layer": layer_name,
    }
    print(output_dict)

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    mlp_str = f"_w_mlp_{args.mlp_dim}" if args.w_mlp else ""
    dropout_str = "_w_dropout" if args.with_dropout else ""
    fname = (
        results_dir
        / f"audiomae_nsynth_{args.task}_{layer_name}_{args.optimizer}_{args.lr}{mlp_str}{dropout_str}.pkl"
    )
    with open(fname, "wb") as f:
        pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to {fname}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layer_idx", type=int, default=11,
                        help="Layer index (0-11 for blocks, 12 for norm).")
    parser.add_argument("--results_dir", default="eval_nsynth_results", type=str)
    parser.add_argument("--model_ckpt_dir", default="model_checkpoints", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--train_epochs", default=20, type=int)
    parser.add_argument("--task", default="family", type=str,
                        choices=["family", "pitch", "instrument", "other"])
    parser.add_argument("--label_field", default=None, type=str)
    parser.add_argument("--nsynth_root", default="/mnt/home/igriffith/ceph/datasets/nsynth", type=str)
    parser.add_argument("--duration", default=None, type=float)
    parser.add_argument("--fade_window_duration", default=0.01, type=float)
    parser.add_argument("--w_mlp", action=BooleanOptionalAction)
    parser.add_argument("--mlp_dim", default=512, type=int)
    parser.add_argument("--with_dropout", action=BooleanOptionalAction)
    parser.add_argument("--lr_scheduler", action=BooleanOptionalAction)
    parser.add_argument("--eval_only", action=BooleanOptionalAction)
    parser.add_argument("--use_classifier_ckpt", action=BooleanOptionalAction)
    args = parser.parse_args()
    cli_main(args)
