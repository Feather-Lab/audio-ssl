"""
Train and evaluate linear probes on frozen AudioMAE representations for
JSIN word recognition and speaker identification.

Mirrors eval_jsin_transfer_matched.py (Whisper path) but uses
AudioMAETransferModule with the pretrained AudioMAE encoder.

Usage:
    python eval_audiomae_jsin_transfer.py \
        --layer_idx 11 --task both --optimizer AdamW --lr 0.0005 \
        --train_epochs 6 --lr_scheduler --with_dropout --no-with_noise
"""

from __future__ import annotations

import os
import sys
import pickle
import pathlib
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from audiomae_transfer_module import AudioMAETransferModule
from jsinV3DataLoader_precombined_batched import CleanSpeechInNoiseValDatasetBatched

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def cli_main(args):
    L.seed_everything(args.random_seed)

    # Build config dict expected by AudioMAETransferModule
    layer_str = f"block_{args.layer_idx}" if args.layer_idx < 12 else "norm"
    config = {
        "model": {
            "arch_name": "audiomae_pretrained",
            "arch_kwargs": {
                "encoder_layer": args.layer_idx,
                "time_average": not args.no_time_avg,
            },
        },
        "hparas": {
            "batch_size": args.batch_size,
            "global_batch_size": int(args.batch_size * args.gpus),
            "optimizer": args.optimizer,
            "lr": args.lr,
            "epochs": args.train_epochs,
        },
        "data": {"eval_max": 3},
        "num_workers": args.num_workers,
        "num_gpus": args.gpus,
        "with_noise": args.with_noise,
        "crop_audio": args.crop_audio,
    }

    if args.lr_scheduler:
        config["hparas"]["lr_schedule"] = True
        config["hparas"]["num_warmup_steps_or_ratio"] = 0

    if args.task == "both":
        config["model"]["arch_kwargs"]["num_classes"] = {
            "signal/word_int": 794,
            "signal/speaker_int": 433,
        }
        config["hparas"]["task_loss_params"] = {
            "signal/word_int": {"loss_type": "crossentropyloss", "weight": 1.0},
            "signal/speaker_int": {"loss_type": "crossentropyloss", "weight": 1.0},
        }
        task_str = "word_and_speaker_task"
    elif args.task == "word":
        config["model"]["arch_kwargs"]["num_classes"] = {"signal/word_int": 794}
        config["hparas"]["task_loss_params"] = {
            "signal/word_int": {"loss_type": "crossentropyloss", "weight": 1.0},
        }
        task_str = "word_task"
    elif args.task == "speaker":
        config["model"]["arch_kwargs"]["num_classes"] = {"signal/speaker_int": 433}
        config["hparas"]["task_loss_params"] = {
            "signal/speaker_int": {"loss_type": "crossentropyloss", "weight": 1.0},
        }
        task_str = "speaker_task"

    config["data"]["target_keys"] = list(config["model"]["arch_kwargs"]["num_classes"].keys())

    if args.w_mlp:
        config["model"]["classifier"] = {"hidden_dims": [args.mlp_dim]}

    if args.with_dropout:
        config["model"]["with_dropout"] = True

    print(f"Running {task_str} transfer on AudioMAE layer {layer_str}")
    print(f"hparas config: {config['hparas']}")

    # Build string modifier for checkpoint / results naming
    time_avg_str = "" if not args.no_time_avg else "full_rep_"
    scheduler_str = "_cosine_lr_scheduler_" if args.lr_scheduler else ""
    mlp_str = "_w_mlp" if args.w_mlp else ""
    dropout_str = "_w_dropout" if args.with_dropout else ""
    noise_str = "_with_noise" if args.with_noise else ""
    crop_str = "_middle_crop" if args.crop_audio else ""

    str_modifier = (
        f"{task_str}_{layer_str}_{time_avg_str}{args.optimizer}_{args.lr}"
        f"{scheduler_str}{mlp_str}{noise_str}{crop_str}{dropout_str}"
    )

    ckpt_dir = pathlib.Path(args.model_ckpt_dir) / f"audiomae/linear_classifier_checkpoints_{str_modifier}"

    module = AudioMAETransferModule(config=config)

    # Load existing classifier checkpoint if available
    classifier_ckpts = list(ckpt_dir.rglob("*.ckpt"))
    if classifier_ckpts and args.use_classifier_ckpt:
        classifier_ckpt_path = str(sorted(classifier_ckpts, key=os.path.getctime)[-1])
        ckpt = torch.load(classifier_ckpt_path, weights_only=False)
        module.load_state_dict(ckpt["state_dict"])
        print(f"Loaded classifier from {classifier_ckpt_path}")

    if args.classifier_ckpt_path:
        ckpt = torch.load(args.classifier_ckpt_path, weights_only=False)
        module.load_state_dict(ckpt["state_dict"])
        print(f"Loaded classifier from {args.classifier_ckpt_path}")

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        ckpt_dir,
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    )
    if args.checkpoint_every_n_steps is not None and args.checkpoint_every_n_steps > 0:
        checkpoint_callback.every_n_train_steps = args.checkpoint_every_n_steps
    callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    wandb_logger = WandbLogger(
        save_dir=str(ckpt_dir),
        name=f"audiomae_classifier_{str_modifier}",
        group="audiomae_transfer",
        project="cochdnn",
    )

    trainer = L.Trainer(
        precision="32",
        default_root_dir=str(pathlib.Path(args.model_ckpt_dir) / "audiomae"),
        max_epochs=config["hparas"]["epochs"],
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp" if args.gpus > 1 else "auto",
        val_check_interval=args.checkpoint_every_n_steps,
        gradient_clip_val=1,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    if not args.eval_only:
        trainer.fit(module)

    # ---- Evaluation ----

    def bootstrap_mean_and_sem(scores, n_bootstraps=1000):
        mean = np.mean(scores)
        boots = [np.mean(np.random.choice(scores, size=len(scores))) for _ in range(n_bootstraps)]
        sem = np.std(boots)
        return mean, sem

    eval_speech_h5_path = (
        "/mnt/home/jfeather/ceph/data/training_datasets_audio/"
        "jsinV3BalancedProcessed/sr_20000/splits/"
        "train_stackedDataframeHDF_n150_VJRUH4IEPDGPNH2JZMULSQKOWYNQ6KMM.pdh5"
    )

    test_dataset = CleanSpeechInNoiseValDatasetBatched(
        speech_h5_path=eval_speech_h5_path,
        target_keys=config["data"]["target_keys"],
        batch_size=config["hparas"]["batch_size"],
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config["num_workers"],
        shuffle=False,
        collate_fn=module.eval_collate_fn,
    )

    print("Running inference")
    outputs = trainer.predict(module, test_dataloader, return_predictions=True)

    top1_word, top1_speaker = [], []
    top5_word, top5_speaker = [], []

    for record in outputs:
        if args.task in ("both", "word"):
            top1_word.append(record["top1"]["signal/word_int"])
            top5_word.append(record["top5"]["signal/word_int"])
        if args.task in ("both", "speaker"):
            top1_speaker.append(record["top1"]["signal/speaker_int"])
            top5_speaker.append(record["top5"]["signal/speaker_int"])

    output_dict = {}
    if args.task in ("both", "word"):
        word_top1_mean, word_top1_sem = bootstrap_mean_and_sem(top1_word)
        word_top5_mean, word_top5_sem = bootstrap_mean_and_sem(top5_word)
        output_dict.update({
            "word_top1_mean": word_top1_mean,
            "word_top1_sem": word_top1_sem,
            "word_top5_mean": word_top5_mean,
            "word_top5_sem": word_top5_sem,
        })
    if args.task in ("both", "speaker"):
        speaker_top1_mean, speaker_top1_sem = bootstrap_mean_and_sem(top1_speaker)
        speaker_top5_mean, speaker_top5_sem = bootstrap_mean_and_sem(top5_speaker)
        output_dict.update({
            "speaker_top1_mean": speaker_top1_mean,
            "speaker_top1_sem": speaker_top1_sem,
            "speaker_top5_mean": speaker_top5_mean,
            "speaker_top5_sem": speaker_top5_sem,
        })

    output_dict = {key: val.item() for key, val in output_dict.items()}
    print(output_dict)

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_filename = results_dir / f"audiomae_linear_eval_jsin_{str_modifier}_center_eval_words.pkl"
    with open(results_filename, "wb") as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to {results_filename}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layer_idx", default=11, type=int, help="ViT block index (0-11). Use 12 for final norm.")
    parser.add_argument("--task", default="both", type=str, help='One of: "both", "word", "speaker".')
    parser.add_argument("--results_dir", default="eval_jsin_results", type=str)
    parser.add_argument("--model_ckpt_dir", default="model_checkpoints", type=str)
    parser.add_argument("--classifier_ckpt_path", default="", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--batch_size", default=192, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--train_epochs", default=6, type=int)
    parser.add_argument("--mlp_dim", default=512, type=int)
    parser.add_argument("--w_mlp", action=BooleanOptionalAction)
    parser.add_argument("--with_noise", action=BooleanOptionalAction)
    parser.add_argument("--with_dropout", action=BooleanOptionalAction)
    parser.add_argument("--eval_only", action=BooleanOptionalAction)
    parser.add_argument("--no_time_avg", action=BooleanOptionalAction, help="If set, do NOT time-average the representation.")
    parser.add_argument("--crop_audio", action=BooleanOptionalAction)
    parser.add_argument("--use_classifier_ckpt", action=BooleanOptionalAction)
    parser.add_argument("--lr_scheduler", action=BooleanOptionalAction)
    parser.add_argument("--checkpoint_every_n_steps", default=None, type=int)
    args = parser.parse_args()
    cli_main(args)
