"""
NSynth Linear Evaluation Script.

Trains linear classifiers on frozen SSL feature extractors for NSynth tasks.
Mirrors eval_jsin_transfer_matched.py structure.
"""

import torch
import torch.nn as nn
import numpy as np
import lightning as L
import yaml
import os
import pickle
import pathlib
from argparse import ArgumentParser, BooleanOptionalAction

from nsynth_linear_eval_module import NSynthLinearEvalModule
from nsynth_dataset import NsynthDataset
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def cli_main(args):
    L.seed_everything(args.random_seed)
    
    # Load config
    if args.config_path != "":
        config_path = pathlib.Path(args.config_path)
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    else:
        raise ValueError("Must provide config_path")
    
    print(f"Evaluating config: {config_path}")
    
    # Update config for transfer learning task
    config['num_workers'] = args.num_workers
    config['num_gpus'] = args.gpus
    config['hparas']['batch_size'] = args.batch_size
    config['hparas']['global_batch_size'] = int(args.batch_size * args.gpus)
    config['hparas']['optimizer'] = args.optimizer
    config['hparas']['lr'] = args.lr
    config['hparas']['epochs'] = args.train_epochs
    
    if 'model' not in config:
        config['model'] = {}
    if 'arch_kwargs' not in config['model']:
        config['model']['arch_kwargs'] = {}
    
    if not args.supervised_backbone:
        config['model']['arch_kwargs']['supervised'] = False
    
    config['model']['arch_kwargs']['time_average'] = args.time_avg_rep
    config['hparas']['lr_schedule'] = args.lr_scheduler
    
    # Get NSynth dataset to determine num_classes
    train_dataset = NsynthDataset(
        nsynth_root=args.nsynth_root,
        split='train',
        task=args.task,
        label_field=args.label_field if args.task == 'other' else None,
        sample_rate=20000,  # Match existing models
        duration=args.duration,  # Duration in seconds (None = full length)
        fade_window_duration=args.fade_window_duration,
    )
    num_classes = train_dataset.num_classes
    print(f"NSynth task '{args.task}': {num_classes} classes")
    
    # Get checkpoint for SSL model
    checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/checkpoints"
    if args.ckpt_path == "":
        ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]  # get latest checkpoint
            print(f"Using checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        ckpt_modifier = ''
    else:
        ckpt_path = args.ckpt_path
        ckpt_modifier = '_from_best_val_ckpt'
    
    # Build modifier string for checkpoint naming
    time_avg_str = ""
    if not args.time_avg_rep:
        time_avg_str = "full_rep_"
    
    scheduler_str = ""
    if args.lr_scheduler:
        scheduler_str = "_cosine_lr_scheduler_"
    
    mlp_str = ""
    if args.w_mlp:
        mlp_str = f"_w_mlp_{args.mlp_dim}"
    
    dropout_str = ""
    if args.with_dropout:
        dropout_str = "_w_dropout"
    
    str_modifier = (
        f"{args.task}_{args.layer_str.replace('.', '_')}_{time_avg_str}"
        f"{config['hparas']['optimizer']}_{config['hparas']['lr']}{scheduler_str}"
        f"{mlp_str}{ckpt_modifier}{dropout_str}"
    )
    
    classifier_checkpoint_dir = (
        pathlib.Path(args.model_ckpt_dir) / 
        f"{config_path.stem}/nsynth_linear_classifier_checkpoints_{str_modifier}"
    )
    
    # Initialize NSynth linear eval module
    module = NSynthLinearEvalModule(
        config=config,
        ckpt_path=ckpt_path,
        layer_out=args.layer_str,
        num_classes=num_classes,
        supervised_backbone=args.supervised_backbone,
        w_mlp=args.w_mlp,
        mlp_dim=args.mlp_dim,
        with_dropout=args.with_dropout,
    )
    
    # Check if existing classifier checkpoint exists
    classifier_ckpts = list(classifier_checkpoint_dir.rglob("*.ckpt"))
    print(f"Existing classifier checkpoints: {classifier_ckpts}")
    classifier_ckpt = None
    
    if len(classifier_ckpts) > 0 and args.use_classifier_ckpt:
        classifier_ckpt_path = str(sorted(classifier_ckpts, key=os.path.getctime)[-1])
        classifier_ckpt = torch.load(classifier_ckpt_path, weights_only=False)
        module.load_state_dict(classifier_ckpt['state_dict'])
        print(f"Loaded classifier from {classifier_ckpt_path}")
    
    if args.classifier_ckpt_path != '':
        classifier_ckpt_path = str(args.classifier_ckpt_path)
        classifier_ckpt = torch.load(classifier_ckpt_path, weights_only=False)
        module.load_state_dict(classifier_ckpt['state_dict'])
        print(f"Loaded classifier from {classifier_ckpt_path}")
    
    # Setup callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(
        classifier_checkpoint_dir,
        monitor="val_acc",  # Monitor validation accuracy for NSynth
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    ))
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup logger
    log_basename = config_path.stem
    wandb_logger = WandbLogger(
        save_dir=checkpoint_dir,
        name=f"{log_basename}_nsynth_{args.task}_classifier_{str_modifier}",
        group='nsynth_linear_eval',
        project='cochdnn'
    )
    
    # Setup trainer
    trainer = L.Trainer(
        precision="32",
        default_root_dir=args.model_ckpt_dir / config_path.stem,
        max_epochs=config['hparas']['epochs'],
        devices=args.gpus,
        accelerator="gpu",
        strategy='ddp' if args.gpus > 1 else 'auto',
        gradient_clip_val=1,
        profiler=None,
        logger=wandb_logger,
        callbacks=callbacks
    )
    
    # Train classifier if not eval_only
    if not args.eval_only:
        trainer.fit(module)
    
    ######################################
    # Run Test Evaluation
    ######################################
    # Create test dataset and dataloader
    test_dataset = NsynthDataset(
        nsynth_root=args.nsynth_root,
        split='test',
        task=args.task,
        label_field=args.label_field if args.task == 'other' else None,
        sample_rate=20000,
        duration=args.duration,
        fade_window_duration=args.fade_window_duration,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    
    print("Running test inference")
    outputs = trainer.predict(module, test_dataloader, return_predictions=True)
    
    # Aggregate results
    top1_scores = []
    top5_scores = []
    
    for record in outputs:
        top1_scores.append(record['top1'])
        top5_scores.append(record['top5'])
    
    n_examples = len(outputs)
    
    output_dict = {
        "top1_mean": torch.stack(top1_scores).mean().item(),
        "top1_sem": torch.stack(top1_scores).std().item() / np.sqrt(n_examples),
        "top5_mean": torch.stack(top5_scores).mean().item(),
        "top5_sem": torch.stack(top5_scores).std().item() / np.sqrt(n_examples),
        "num_classes": num_classes,
        "task": args.task,
        "layer": args.layer_str,
    }
    
    print(output_dict)
    
    # Save results as .pkl
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_filename = (
        results_dir / 
        f"{config_path.stem}_nsynth_{args.task}_linear_eval_{str_modifier}.pkl"
    )
    with open(results_filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Results saved to {results_filename}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--config_path',
        default='',
        type=str,
        help='Path to experiment config YAML file.'
    )
    parser.add_argument(
        "--results_dir",
        default=pathlib.Path("./eval_nsynth_results"),
        type=pathlib.Path,
        help="Directory where model results will be saved. (Default: './eval_nsynth_results')",
    )
    parser.add_argument(
        "--model_ckpt_dir",
        default=pathlib.Path("./model_checkpoints"),
        type=pathlib.Path,
        help="Directory where model checkpoints exist. (Default: './model_checkpoints')",
    )
    parser.add_argument(
        "--ckpt_path",
        default='',
        type=str,
        help="Path to specific checkpoint (defaults to latest in checkpoint dir)."
    )
    parser.add_argument(
        "--classifier_ckpt_path",
        default='',
        type=str,
        help="Path to specific classifier checkpoint to load."
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs per node to use. (Default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size to use. (Default: 256)",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of CPUs for dataloader. (Default: 4)",
    )
    parser.add_argument(
        '--random_seed',
        default=0,
        type=int,
        help='Random seed'
    )
    parser.add_argument(
        '--layer_str',
        default='avgpool',
        type=str,
        help='Layer to fit classifier on top of (e.g., "avgpool", "layer4").'
    )
    parser.add_argument(
        '--task',
        default='family',
        type=str,
        choices=['family', 'pitch', 'instrument', 'other'],
        help='NSynth task: family (11 classes), pitch (128 classes), instrument (variable), or other.'
    )
    parser.add_argument(
        '--label_field',
        default=None,
        type=str,
        help='Metadata field to use as label when task="other".'
    )
    parser.add_argument(
        '--nsynth_root',
        default='/mnt/home/igriffith/ceph/datasets/nsynth',
        type=str,
        help='Root directory containing NSynth dataset.'
    )
    parser.add_argument(
        '--optimizer',
        default='LARS',
        type=str,
        help='Optimizer to use (e.g., "LARS", "Adam", "SGD").'
    )
    parser.add_argument(
        '--lr',
        default=0.2,
        type=float,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--w_mlp',
        action=BooleanOptionalAction,
        help='Use MLP instead of linear classifier?'
    )
    parser.add_argument(
        '--mlp_dim',
        default=512,
        type=int,
        help='Hidden dimension of MLP (if w_mlp is True).'
    )
    parser.add_argument(
        '--with_dropout',
        action=BooleanOptionalAction,
        help='Include dropout layer in classifier?'
    )
    parser.add_argument(
        '--eval_only',
        action=BooleanOptionalAction,
        help='Only evaluate using existing classifier?'
    )
    parser.add_argument(
        '--time_avg_rep',
        action=BooleanOptionalAction,
        default=True,
        help='Time average the model representation fed to classifier?'
    )
    parser.add_argument(
        '--use_classifier_ckpt',
        action=BooleanOptionalAction,
        help='Use existing classifier checkpoint?'
    )
    parser.add_argument(
        '--lr_scheduler',
        action=BooleanOptionalAction,
        help='Use learning rate scheduler?'
    )
    parser.add_argument(
        '--supervised_backbone',
        action=BooleanOptionalAction,
        help='Using supervised backbone?'
    )
    parser.add_argument(
        '--train_epochs',
        default=20,
        type=int,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--duration',
        default=None,
        type=float,
        help='Duration in seconds to crop/pad audio (None = use full length).'
    )
    parser.add_argument(
        '--fade_window_duration',
        default=0.01,
        type=float,
        help='Duration in seconds for Hann window fade-out at right edge (default: 0.01 = 10ms).'
    )
    
    args = parser.parse_args()
    
    cli_main(args)

