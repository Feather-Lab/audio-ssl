import torch 
import torch.nn as nn 
import numpy as np
import lightning as L
import yaml
import sys
import os
import pickle
import pathlib
from argparse import ArgumentParser, BooleanOptionalAction

from lightning_ssl_classifier import SSLClassifier
from lightning_byola_classifier import BYOLAClassifier
from whisper_encoder_arch import get_whisper_encoder_layer_map
from jsinV3DataLoader_precombined_batched import CleanSpeechInNoiseValDatasetBatched
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def cli_main(args):
    L.seed_everything(args.random_seed)
    
    # Load config
    if args.config_path != "":
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
        raise ValueError("Must provide either config_path or config_list_path")

    print(f"Evaluating config: {config_path}")
    
    # Determine which model type to use
    use_whisper = False
    use_byola = False
    # Only treat as pretrained whisper when explicitly requested.
    use_whisper_pretrained = args.model_type == 'whisper'
    
    if use_whisper_pretrained:
        use_whisper = True
    elif config.get('model', {}).get('arch_kwargs', {}).get('backbone') == 'whisper':
        use_whisper = True
    elif 'whisper' in config.get('model', {}).get('arch_name', ''):
        use_whisper = True
    
    whisper_layer_str = None
    if use_whisper:
        if 'model' not in config:
            config['model'] = {}
        if 'hparas' not in config:
            config['hparas'] = {}
        if 'data' not in config:
            config['data'] = {}
        if 'arch_kwargs' not in config['model']:
            config['model']['arch_kwargs'] = {}

        encoder_kwargs = config['model']['arch_kwargs'].get('encoder_kwargs', {})
        n_layer = encoder_kwargs.get('n_layer', 4)
        layer_name_map = get_whisper_encoder_layer_map(n_layer)

        if args.print_whisper_layers:
            print(f"Whisper encoder layers (n_layer={n_layer}):")
            for key in sorted(layer_name_map.keys()):
                print(f"{key} -> {layer_name_map[key]}")
            sys.exit(0)

        if args.layer_str.isdigit():
            layer_idx = int(args.layer_str)
            if layer_idx < 0 or layer_idx >= n_layer:
                raise ValueError(
                    f"Whisper encoder has {n_layer} layers; got index {layer_idx}."
                )
            layer_key = f"encoder_block_{layer_idx}"
        else:
            layer_key = args.layer_str

        if layer_key not in layer_name_map:
            valid_names = ", ".join(sorted(layer_name_map.keys()))
            raise ValueError(
                f"Invalid whisper layer_str '{args.layer_str}'. "
                f"Use an integer block index or one of: {valid_names}"
            )

        whisper_layer_str = layer_name_map[layer_key]
        config['model']['arch_kwargs']['time_average'] = (
            False if args.time_avg_rep is None else args.time_avg_rep
        )
    elif 'byol-a' in str(config_path):
        # init model and hparas dicts  for byola configs 
        config['model'] = {}
        config['hparas'] = {}
        # init audio transforms and model arch kwargs for byola configs 
        config['audio_transforms'] = {} 
        config['audio_transforms']['low_snr'] = -10
        config['audio_transforms']['high_snr'] = 10
        config['audio_transforms']['rms_level'] = 60
        config['model']['arch_kwargs'] = {}
        config['data'] = {}
        config['classifier_layer'] = args.layer_str
        use_byola = True 

    # update config for transfer learning task
    config['num_workers'] = args.num_workers
    config['num_gpus'] = args.gpus
    config['hparas']['batch_size'] = args.batch_size
    config['hparas']['global_batch_size'] = int(args.batch_size * args.gpus)
    config['data']['eval_max'] = 3
    config['hparas']['optimizer'] = args.optimizer
    # used 2 gpus for training, mult by 2 for now to get same checkpoint 
    if not use_whisper:  # Whisper uses the SSL encoder backbone
        if 'model' not in config:
            config['model'] = {}
        if 'arch_kwargs' not in config['model']:
            config['model']['arch_kwargs'] = {}
        if not args.supervised_backbone:
            config['model']['arch_kwargs']['supervised'] = False

    config['with_noise'] = args.with_noise
    # don't load in classifier head if it exists 
    config['hparas']['lr'] = args.lr 
    config['hparas']['epochs'] = args.train_epochs
    if 'arch_kwargs' not in config['model'].keys():
        config['model']['arch_kwargs'] = {}
    if not use_whisper:
        config['model']['arch_kwargs']['time_average'] = args.time_avg_rep
    config['crop_audio'] = args.crop_audio

    crop_audio_str = ""
    if  args.crop_audio:
        crop_audio_str = "_middle_crop"

    time_avg_str = ""
    if not args.time_avg_rep:
        time_avg_str = "full_rep_"

    scheduler_str = ""

    if args.lr_scheduler:
        config['hparas']['lr_schedule'] = True
        config['hparas']['num_warmup_steps_or_ratio'] = 0
        scheduler_str = "_cosine_lr_scheduler_"

    if args.task == 'both':
        # add to model arch kwargs to add classifier head 
        config['model']['arch_kwargs']['num_classes'] = {"signal/word_int": 794,    
                                    "signal/speaker_int": 433} 
        task_str = f"word_and_speaker_task"
        # add task loss params to hparas 
        config['hparas']['task_loss_params'] = {
            "signal/word_int":
                {"loss_type": 'crossentropyloss',
                "weight": 1.0},                                       # init loss is ~200 
            "signal/speaker_int":
                {"loss_type": 'crossentropyloss',
                "weight": 1.0}
            }
        
    elif args.task == 'word':
        config['model']['arch_kwargs']['num_classes'] = {"signal/word_int": 794} 
        task_str = f"word_task"
        # add task loss params to hparas 
        config['hparas']['task_loss_params'] = {
            "signal/word_int":
                {"loss_type": 'crossentropyloss',
                "weight": 1.0}
            }

    elif args.task == 'speaker':
        config['model']['arch_kwargs']['num_classes'] = {"signal/speaker_int": 433} 
        task_str = f"speaker_task"
        # add task loss params to hparas 
        config['hparas']['task_loss_params'] = {
            "signal/speaker_int":
                {"loss_type": 'crossentropyloss',
                "weight": 1.0}
            }
    

    ## update target keys 
    config['data']['target_keys'] = list(config['model']['arch_kwargs']['num_classes'].keys())
    print(f"Running {task_str} transfer")
    
    print(f"hparas config: {config['hparas']}")
    print(f"model config: {config['model']['arch_kwargs']}")

    if args.w_mlp:
        config['model']['classifier'] = {}
        config['model']['classifier']['hidden_dims'] = [args.mlp_dim]
        mlp_str = "_w_mlp"
    else:
        mlp_str = ""
    
    if args.with_dropout:
        config['model']['with_dropout'] = True
        dropout_str = "_w_dropout"
    else:
        dropout_str = ""


    # get checkpoint for ssl model 
    checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/checkpoints"
    if args.ckpt_path == "":
        ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1] # get latest checkpoint 
            print(ckpt_path)
        ckpt_modifier = ''

    else:
        ckpt_path = args.ckpt_path
        ckpt_modifier = '_from_best_val_ckpt'
    
    w_noise_modifier = '_with_noise' if args.with_noise else ""
    
    layer_component = whisper_layer_str if use_whisper else args.layer_str.replace('.', '_')
    str_modifier = f"{task_str}_{layer_component}_{time_avg_str}{config['hparas']['optimizer']}_{config['hparas']['lr']}{scheduler_str}{mlp_str}{ckpt_modifier}{w_noise_modifier}{crop_audio_str}{dropout_str}"
    classifier_checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/linear_classifier_checkpoints_{str_modifier}"

    if use_byola:
        module = BYOLAClassifier(config=config)
    else:
        module = SSLClassifier(config=config,
                            ckpt_path=ckpt_path,
                            layer_out=whisper_layer_str if use_whisper else args.layer_str,
                            supervised_backbone=args.supervised_backbone)

    ## Check if existing classifier_ckpt exists 
    classifier_ckpts = list(classifier_checkpoint_dir.rglob("*.ckpt"))
    print(f"Existing classifier checkpoints: {classifier_ckpts}")
    classifier_ckpt = None 

    if len(classifier_ckpts) > 0 and args.use_classifier_ckpt:
        classifier_ckpt_path = str(sorted(classifier_ckpts, key=os.path.getctime)[-1])
        classifier_ckpt = torch.load(classifier_ckpt_path, weights_only=False) # get latest checkpoint 
        module.load_state_dict(classifier_ckpt['state_dict'])
        print(f"Loaded classifier from {classifier_ckpt_path}")
    
    if args.classifier_ckpt_path != '':
        classifier_ckpt_path = str(args.classifier_ckpt_path)
        classifier_ckpt = torch.load(classifier_ckpt_path, weights_only=False) # get latest checkpoint 
        module.load_state_dict(classifier_ckpt['state_dict'])
        print(f"Loaded classifier from {classifier_ckpt_path}")
        

    callbacks=[]
    checkpoint_callback = ModelCheckpoint(
        classifier_checkpoint_dir,
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    )
    # Add incremental checkpointing if specified
    if args.checkpoint_every_n_steps is not None and args.checkpoint_every_n_steps > 0:
        checkpoint_callback.every_n_train_steps = args.checkpoint_every_n_steps
    callbacks.append(checkpoint_callback)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    # callbacks.append(EarlyStopping(monitor="train_classifier_loss", mode="min"))

    if use_byola:
        log_basename = 'byol-a_base'
    else:
        log_basename = config_path.stem

    wandb_logger = WandbLogger(save_dir=checkpoint_dir, 
                               name=f"{log_basename}_classifier_{str_modifier}",
                               group='word_classifier_transfer',
                               project='cochdnn')

    trainer = L.Trainer(
        precision="32",
        # limit_val_batches=0,
        default_root_dir=args.model_ckpt_dir / config_path.stem,
        max_epochs=config['hparas']['epochs'],
        devices=args.gpus,
        accelerator="gpu", 
        strategy='ddp' if args.gpus > 1 else 'auto',
        val_check_interval = args.checkpoint_every_n_steps, 
        # limit_train_batches=2,
        # limit_val_batches=2,
        gradient_clip_val=1, # clipt grad l2 norm to 1 
        profiler=None,
        logger=wandb_logger,
        callbacks=callbacks)   
    
    # train classifier if we haven't already, or if overwriting 
    if not args.eval_only:
        # if not classifier_ckpt or args.overwrite_classifier:
            # fit classifier 
        trainer.fit(module)

    ######################################
    # Run Eval
    ######################################
    def eval_collate_fn(batch):
        audio, targets = batch[0] # unbox wrapper added by dataloader 
        audio = audio.unsqueeze(1)
        # # combine labels: each target is dict for each key, stack the values 
        labels = {}
        for label_key in targets.keys():
            labels[label_key] = torch.from_numpy(targets[label_key])
        return audio, labels

    if use_whisper_pretrained:
        eval_collate_fn = module.eval_collate_fn
    elif 'byol-a' in str(config_path):
        eval_collate_fn = module.predict_collate_fn
    else:
        eval_collate_fn = eval_collate_fn

    eval_speech_h5_path = '/mnt/home/jfeather/ceph/data/training_datasets_audio/jsinV3BalancedProcessed/sr_20000/splits/train_stackedDataframeHDF_n150_VJRUH4IEPDGPNH2JZMULSQKOWYNQ6KMM.pdh5'

    test_dataset = CleanSpeechInNoiseValDatasetBatched(speech_h5_path=eval_speech_h5_path,
                                            target_keys=config['data']['target_keys'],
                                            batch_size=config['hparas']['batch_size'],
                                            )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config['num_workers'],
        shuffle=False,
        collate_fn=eval_collate_fn
    )

    print("Running inference")
    outputs = trainer.predict(module, test_dataloader, return_predictions=True)
    # get stats from test 
    top1_word = []
    top1_speaker = []
    top5_word = []
    top5_speaker = []

    for record in outputs:
        if args.task == 'both' or args.task == 'word':
            top1_word.append(record['top1']['signal/word_int'])
            top5_word.append(record['top5']['signal/word_int'])
        if args.task == 'both' or args.task == 'speaker':
            top1_speaker.append(record['top1']['signal/speaker_int'])
            top5_speaker.append(record['top5']['signal/speaker_int'])
    n_examples = len(outputs)
    
    if args.task == 'both':
        output_dict = {
            "word_top1_mean": torch.stack(top1_word).mean(),
            "word_top1_sem": torch.stack(top1_word).std() / np.sqrt(n_examples),
            "speaker_top1_mean": torch.stack(top1_speaker).mean(),
            "speaker_top1_sem": torch.stack(top1_speaker).std() / np.sqrt(n_examples),

            "word_top5_mean": torch.stack(top5_word).mean(),
            "word_top5_sem": torch.stack(top5_word).std() / np.sqrt(n_examples),
            "speaker_top5_mean": torch.stack(top5_speaker).mean(),
            "speaker_top5_sem": torch.stack(top5_speaker).std() / np.sqrt(n_examples),
        }
    elif args.task == 'word':
        output_dict = {
            "word_top1_mean": torch.stack(top1_word).mean(),
            "word_top1_sem": torch.stack(top1_word).std() / np.sqrt(n_examples),
            "word_top5_mean": torch.stack(top5_word).mean(),
            "word_top5_sem": torch.stack(top5_word).std() / np.sqrt(n_examples),
        }
    elif args.task == 'speaker':
        output_dict = {
            "speaker_top1_mean": torch.stack(top1_speaker).mean(),
            "speaker_top1_sem": torch.stack(top1_speaker).std() / np.sqrt(n_examples),
            "speaker_top5_mean": torch.stack(top5_speaker).mean(),
            "speaker_top5_sem": torch.stack(top5_speaker).std() / np.sqrt(n_examples),
        }
        
    output_dict = {key:val.item() for key,val in output_dict.items()}  

    print(output_dict)
    # save results as .pkl 
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_filename = results_dir / f"{config_path.stem}_linear_eval_jsin_{str_modifier}_center_eval_words.pkl"
    with open(results_filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                       

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='', type=str, help='Path to experiment config.')
    parser.add_argument('--config_list_path', default='', type=str, help='Path to experiment config.')
    parser.add_argument('--model_type', default='', type=str, help='Model type: "whisper", "byola", or "ssl" (auto-detected if not specified).')
    parser.add_argument(
        "--results_dir",
        default=pathlib.Path("./eval_jsin_results"),
        type=pathlib.Path,
        help="Directory where model results will be saved. (Default: './eval_jsin_results')",
    )
    parser.add_argument(
        "--model_ckpt_dir",
        default=pathlib.Path("./model_checkpoints"),
        type=pathlib.Path,
        help="Directory where model checkpoints exists. (Default: './model_checkpoints')",
    )
    parser.add_argument(
        "--ckpt_path",
        default='',
        type=str,
        help="Test from this checkpoint."
    )
    parser.add_argument(
        "--classifier_ckpt_path",
        default='',
        type=str,
        help="Test from this checkpoint."
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs per node to use for test. (Default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size to use for test. (Default: 256)",
    )
    parser.add_argument(
    "--num_workers",
    default=0,
    type=int,
    help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument('--random_seed', default=0, type=int, help='Random seed')
    parser.add_argument('--layer_str', default='avgpool', type=str, help='Layer to fit classifier on top of. For whisper encoders, use "encoder_block_0", "conv1", "ln_post", or a block index.')
    parser.add_argument('--task', default='both', type=str, help='One of: ["both", "word", "speaker"]. Default is "both"')
    parser.add_argument('--optimizer', default='LARS', type=str, help='String for optimizer used.')
    parser.add_argument('--lr', default=0.2, type=float, help='Initial LR used.')
    parser.add_argument('--w_mlp', action=BooleanOptionalAction, help='Use MLP instead of linear classifier?')
    parser.add_argument('--with_noise', action=BooleanOptionalAction, help='Include noise in training?')
    parser.add_argument('--with_dropout', action=BooleanOptionalAction, help='Include dropout layer in classifier?')
    parser.add_argument('--overwrite_classifier', action=BooleanOptionalAction, help='Overwrite existing classifer?')
    parser.add_argument('--eval_only', action=BooleanOptionalAction, help='Eval using existing classifer?')
    parser.add_argument('--time_avg_rep', action=BooleanOptionalAction, help='Time average the model rep fed to classifer?')
    parser.add_argument('--crop_audio', action=BooleanOptionalAction, help='Randomly crop audio to 1 second, centered on word?')
    parser.add_argument('--use_classifier_ckpt', action=BooleanOptionalAction, help='Use existing classifer ckpt?')
    parser.add_argument('--mlp_dim', default=512, type=int, help='Hidden dim of MLP.')
    parser.add_argument('--lr_scheduler', action=BooleanOptionalAction, help='Use lr scheduler?')
    parser.add_argument('--supervised_backbone', action=BooleanOptionalAction, help='Using supervised backbone?')
    parser.add_argument('--array_ix', default=0, type=int, help='Slurm job array index')
    parser.add_argument('--train_epochs', default=3, type=int, help='Number of training epochs.')
    parser.add_argument('--print_whisper_layers', action=BooleanOptionalAction, help='Print Whisper encoder layer map and exit.')
    parser.add_argument(
        '--checkpoint_every_n_steps',
        default=None,
        type=int,
        help='Checkpoint every N training steps (incremental checkpointing). If None, only checkpoint at end of epochs. (Default: None)'
    )
    args = parser.parse_args()

    cli_main(args)
