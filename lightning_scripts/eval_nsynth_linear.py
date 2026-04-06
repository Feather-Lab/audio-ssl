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


def _parse_audiomae_layer_str(layer_str: str) -> tuple:
    """Parse a layer_str into (encoder_layer_idx, layer_name) for AudioMAE."""
    if layer_str == "norm":
        return 12, "norm"
    if layer_str.startswith("block_"):
        idx = int(layer_str.split("_")[-1])
        return idx, layer_str
    if layer_str.isdigit():
        idx = int(layer_str)
        return idx, f"block_{idx}" if idx < 12 else "norm"
    raise ValueError(
        f"Invalid AudioMAE layer_str '{layer_str}'. "
        "Use 'block_N' (0-11), 'norm', or an integer."
    )


def cli_main(args):
    L.seed_everything(args.random_seed)
    
    # Load config
    if args.encoder_type == 'audiomae':
        audiomae_layer_idx, audiomae_layer_name = _parse_audiomae_layer_str(args.layer_str)
        config = {
            'model': {
                'arch_name': 'audiomae_pretrained',
                'arch_kwargs': {
                    'encoder_layer': audiomae_layer_idx,
                    'time_average': True,
                },
            },
            'hparas': {},
        }
        config_path = pathlib.Path('audiomae_pretrained')
    elif args.config_path != "":
        config_path = pathlib.Path(args.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    elif args.config_list_path != "":
        with open(args.config_list_path, 'rb') as f:
            config_dict = pickle.load(f)
            config_path = pathlib.Path(config_dict[args.array_ix])
            config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    else:
        raise ValueError("Must provide either config_path, config_list_path, or --encoder_type audiomae")

    use_audiomae = args.encoder_type == 'audiomae'
    use_whisper = args.encoder_type == 'whisper'
    if not use_whisper and not use_audiomae and 'whisper' in str(config_path).lower():
        use_whisper = True
    
    print(f"Evaluating config: {config_path}")
    
    # Check if this is BYOL-A (pre-trained, no checkpoint needed)
    use_byola = not use_audiomae and (args.encoder_type == 'byola' or 'byol-a' in str(config_path).lower() or 'byola' in str(config_path).lower())
    
    # Infer supervised backbone from config path if not explicitly set
    supervised_backbone = args.supervised_backbone
    if supervised_backbone is None and 'supervised_models' in str(config_path):
        supervised_backbone = True
        print(f"Inferred supervised_backbone=True from config path")
    elif supervised_backbone is None:
        supervised_backbone = False
    
    # Handle BYOL-A config setup (same as speech_commands) - must happen before accessing config values
    if use_byola:
        config['model'] = {}
        config['hparas'] = {}
        config['model']['arch_kwargs'] = {}
    
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
    
    # Handle encoder configuration
    if use_audiomae:
        pass  # config already built above
    elif use_whisper:
        if 'arch_kwargs' not in config['model']:
            config['model']['arch_kwargs'] = {}
        if 'whisper_model' not in config['model']:
            config['model']['whisper_model'] = args.whisper_model
        try:
            encoder_layer = eval(args.layer_str)
            if not isinstance(encoder_layer, int):
                encoder_layer = int(args.layer_str)
        except (ValueError, SyntaxError, NameError):
            try:
                encoder_layer = int(args.layer_str)
            except ValueError:
                raise ValueError(f"layer_str must be convertible to int when using Whisper, got {args.layer_str}")
        config['model']['arch_kwargs']['encoder_layer'] = encoder_layer
    else:
        # Handle supervised models (use arch_params) vs SSL models (use arch_kwargs)
        if supervised_backbone:
            # Supervised models use arch_params for checkpoint loading, but SSLClassifier needs arch_kwargs
            # Keep both: arch_params for model loading, arch_kwargs for SSLClassifier internal logic
            if 'arch_params' in config['model']:
                if 'arch_kwargs' not in config['model']:
                    config['model']['arch_kwargs'] = {}
                # Copy arch_params to arch_kwargs (but keep arch_params for checkpoint loading)
                import copy
                for key, value in config['model']['arch_params'].items():
                    if key not in config['model']['arch_kwargs']:
                        # Deep copy to avoid modifying the original arch_params
                        if isinstance(value, dict):
                            config['model']['arch_kwargs'][key] = copy.deepcopy(value)
                        else:
                            config['model']['arch_kwargs'][key] = value
                # Extract backbone from arch_name for supervised models
                arch_name = config['model'].get('arch_name', '')
                if 'kell2018' in arch_name:
                    config['model']['arch_kwargs']['backbone'] = 'kell2018'
                elif 'resnet' in arch_name:
                    # Extract resnet type from arch_name
                    if 'resnet18' in arch_name or 'resnet_multi_task18' in arch_name:
                        config['model']['arch_kwargs']['backbone'] = 'resnet18'
                    else:
                        config['model']['arch_kwargs']['backbone'] = 'resnet50'
            else:
                # If no arch_params, create arch_kwargs from scratch
                if 'arch_kwargs' not in config['model']:
                    config['model']['arch_kwargs'] = {}
        else:
            # SSL models use arch_kwargs
            if 'arch_kwargs' not in config['model']:
                config['model']['arch_kwargs'] = {}
            config['model']['arch_kwargs']['supervised'] = False
        config['model']['arch_kwargs']['time_average'] = args.time_avg_rep
    config['hparas']['lr_schedule'] = args.lr_scheduler
    
    # AudioMAE, BYOL-A, and Whisper use 16kHz; SSL models use 20kHz
    target_sample_rate = 16000 if (use_audiomae or use_whisper or use_byola) else 20000

    # Get NSynth dataset to determine num_classes
    train_dataset = NsynthDataset(
        nsynth_root=args.nsynth_root,
        split='train',
        task=args.task,
        label_field=args.label_field if args.task == 'other' else None,
        sample_rate=target_sample_rate,
        duration=args.duration,  # Duration in seconds (None = full length)
        fade_window_duration=args.fade_window_duration,
    )
    num_classes = train_dataset.num_classes
    print(f"NSynth task '{args.task}': {num_classes} classes")
    
    # Set num_classes in config for SSLClassifier
    config['model']['arch_kwargs']['num_classes'] = num_classes
    
    # Set task_loss_params for single-task NSynth classification
    # SSLClassifier requires task_loss_params even for single-task scenarios
    task_key = f"nsynth/{args.task}"
    if 'hparas' not in config:
        config['hparas'] = {}
    config['hparas']['task_loss_params'] = {
        task_key: {
            'loss_type': 'crossentropyloss',
            'weight': 1.0
        }
    }
    
    # Build modifier string for checkpoint naming (same pattern as speech_commands)
    time_avg_str = ""
    if not use_whisper and not use_audiomae and not args.time_avg_rep:
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
    
    # Get checkpoint for encoder if needed (same pattern as speech_commands)
    checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/checkpoints"
    if use_audiomae:
        ckpt_path = None
        ckpt_modifier = ''
    elif use_whisper:
        ckpt_path = None
        ckpt_modifier = '_whisper'
    elif use_byola:
        # For pre-trained BYOL-A, no checkpoint is needed
        ckpt_path = None
        ckpt_modifier = ''
    else:
        if args.ckpt_path == "":
            ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
            if len(ckpt_paths) > 0:
                ckpt_path = ckpt_paths[-1]  # get latest checkpoint
                print(ckpt_path)
            ckpt_modifier = ''
        else:
            ckpt_path = args.ckpt_path
            ckpt_modifier = '_from_best_val_ckpt'
    
    if use_audiomae:
        layer_component = audiomae_layer_name
    elif use_whisper:
        layer_component = f"whisper_layer_{config['model']['arch_kwargs']['encoder_layer']}"
    else:
        layer_component = args.layer_str.replace('.', '_')
    str_modifier = (
        f"{config['hparas']['optimizer']}_{layer_component}_{time_avg_str}"
        f"{config['hparas']['lr']}{scheduler_str}{mlp_str}{ckpt_modifier}{dropout_str}"
    )
    
    classifier_checkpoint_dir = (
        pathlib.Path(args.model_ckpt_dir) / 
        f"{config_path.stem}/nsynth_linear_classifier_checkpoints/{str_modifier}"
    )
    
    # Initialize NSynth linear eval module
    module = NSynthLinearEvalModule(
        config=config,
        ckpt_path=ckpt_path,
        layer_out=args.layer_str,
        num_classes=num_classes,
        supervised_backbone=supervised_backbone,
        w_mlp=args.w_mlp,
        mlp_dim=args.mlp_dim,
        with_dropout=args.with_dropout,
        nsynth_root=args.nsynth_root,
        task=args.task,
        label_field=args.label_field if args.task == 'other' else None,
        duration=args.duration,
        fade_window_duration=args.fade_window_duration,
        sample_rate=target_sample_rate,
        use_whisper=use_whisper,
        use_byola=use_byola,
        use_audiomae=use_audiomae,
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
        sample_rate=target_sample_rate,
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
    
    # Bootstrap function for computing SEM (same as eval_jsin_transfer_matched.py)
    def bootstrap_mean_and_sem(scores, n_bootstraps=1000):
        mean = np.mean(scores)
        boots = [np.mean(np.random.choice(scores, size=len(scores))) for _ in range(n_bootstraps)]
        # sem is std of the bootstraps
        sem = np.std(boots)
        return mean, sem
    
    # Aggregate results
    top1_scores = []
    top5_scores = []
    
    for record in outputs:
        top1_scores.append(record['top1'].item())
        top5_scores.append(record['top5'].item())
    
    # Compute mean and SEM via bootstrapping
    top1_mean, top1_sem = bootstrap_mean_and_sem(top1_scores, n_bootstraps=1000)
    top5_mean, top5_sem = bootstrap_mean_and_sem(top5_scores, n_bootstraps=1000)
    
    output_dict = {
        "top1_mean": top1_mean,
        "top1_sem": top1_sem,
        "top5_mean": top5_mean,
        "top5_sem": top5_sem,
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
        '--config_list_path',
        default='',
        type=str,
        help='Path to pickle file containing dict mapping array indices to config paths.'
    )
    parser.add_argument(
        '--array_ix',
        default=0,
        type=int,
        help='Slurm job array index (used with config_list_path).'
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
    parser.add_argument(
        '--encoder_type',
        default='ssl',
        choices=['ssl', 'whisper', 'audiomae', 'byola'],
        help='Encoder to use for feature extraction. Set to "whisper", "audiomae", or "byola" for pretrained encoders.'
    )
    parser.add_argument(
        '--whisper_model',
        default='large-v3-turbo',
        type=str,
        help='Whisper model name when encoder_type="whisper" (e.g., "large-v3-turbo").'
    )
    
    args = parser.parse_args()
    
    cli_main(args)

