import os 
import torch 
import yaml 
import pickle 
import pathlib
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

#TODO: Make below a module dict that can import right lightning module from config 
from lightning_classifier import LitWordAudioSetModel
from lightning_classifier_matched_speech_in_noise import LitWordAudioSetModel as LitWordAudioSetModelMatched
from lightning_ssl import LitAudioSSL 
from lightning_ssl_sep_classifier_opt import LitAudioSSL as LitAudioSSLSepClassOpt
from lightning_ssl_matched_speech_in_noise import LitAudioSSL as LitAudioSSLMatched
from lightning_ssl_matched_audioset_only import LitAudioSSL as LitAudioSSLMatchedAudioset
from lightning_ssl_audioset import LitAudioSetSSL

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins.environments import SLURMEnvironment

import logging
logging.getLogger('sox').setLevel(logging.ERROR)

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def cli_main(args):
    L.seed_everything(args.random_seed)


    if args.config_path != "":
        config_path = pathlib.Path(args.config_path)


    else:
        with open(args.config_list, 'rb') as f:
            model_config = pickle.load(f)
        config_path = pathlib.Path(model_config[args.array_id])

    print(config_path)

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    # set num_workers from cl args as total workers // gpus 
    config['num_workers'] = (args.num_workers // args.gpus) - 1 # let 1 worker handle the process for each gpu
    config['num_gpus'] = args.gpus
    # set batch size per task as global_batch // gpus 

    # set checkpoint dir 
    checkpoint_dir = args.exp_dir / f"{config_path.stem}/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # get task-specific inits 
    classifier = False
    if any(task_str in config_path.stem for task_str in ['ssl', 'barlow', 'mmcr']):
        if config['hparas'].get('sep_class_opt', False):
            module = LitAudioSSLSepClassOpt

        elif config['data'].get('dataset', False) == "MatchedSpeechInNoiseDatasetBatched":
            module = LitAudioSSLMatched

        elif config['data'].get('dataset', False) == "MatchedAudiosetBatched":
            module = LitAudioSSLMatchedAudioset

        elif config.get('module', None):
            module =  eval(config['module'])

        else:
            module = LitAudioSSL
        config['hparas']['batch_size'] = config['hparas']['global_batch_size'] // args.gpus

    else:
        classifier = True
        dataset_name = config['data'].get('dataset', "")
        if dataset_name in ["MatchedSpeechInNoiseDatasetBatched", "MatchedAudiosetBatched"]:
            module = LitWordAudioSetModelMatched
        else:
            module = LitWordAudioSetModel
        config['hparas']['batch_size'] = config['hparas']['batch_size'] // args.gpus

    val_metrics = config.get("val_metric", None)
    callbacks = []
    if val_metrics: 
        if isinstance(config['val_metric'], dict):
            for name, value in config['val_metric'].items():
                callbacks.append(ModelCheckpoint(
                    checkpoint_dir,
                    filename="{epoch}-{step}-best_"+name,
                    monitor=value,
                    mode="max",
                    save_top_k=1,
                    # save_weights_only=True,
                    verbose=True,
                ))
        else:
            callbacks.append(ModelCheckpoint(
                checkpoint_dir,
                monitor=f"val_{config['val_metric']}",
                mode="max" if 'acc' in config['val_metric'] else "min",
                filename=f"{epoch}-{step}-best_{config['val_metric']}",
                save_top_k=1,
                save_weights_only=False,
                verbose=True,
            ))
                    
            
    if classifier:
        callbacks.append(ModelCheckpoint(
            checkpoint_dir,
            monitor="train_loss",
            filename="{epoch}-{step}-best_train",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
            verbose=True,
        ))
        callbacks.append(ModelCheckpoint(
            checkpoint_dir,
            monitor="val_loss",
            filename="{epoch}-{step}-best_train",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
            verbose=True,
        ))
        val_loss_to_stop_on = "val_loss"

    else:
        callbacks.append(ModelCheckpoint(
            checkpoint_dir,
            monitor="train_total_loss",
            filename="{epoch}-{step}-best_train",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
            verbose=True,
        ))
        callbacks.append(ModelCheckpoint(
            checkpoint_dir,
            monitor='val_total_loss',
            filename="{epoch}-{step}-best_val",
            mode= "min",
            save_top_k=1,
            save_weights_only=False,
            verbose=True,
        ))
        val_loss_to_stop_on = "val_total_loss"


    ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
    ckpt_path = None if args.ckpt_path == '' else args.ckpt_path
    if args.resume_training and (len(ckpt_paths) != 0 or ckpt_path):
        ckpt_path = ckpt_paths[-1] if ckpt_path is None else ckpt_path 
        model = module.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)
        print('Resuming training from checkpoint: ', ckpt_path)
    else:
        model = module(config)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    wandb_logger = WandbLogger(save_dir=checkpoint_dir, 
                               name=config_path.stem,
                               group='supervised_models' if classifier else "SSL_models",
                               project='cochdnn')
    # log gradients 
    wandb_logger.watch(model.model, log="all",)
    # Early stopping to save compute
    # if val loss does not improve by 0.1 for 3 epochs (default), end training 
    # callbacks.append(EarlyStopping(monitor=val_loss_to_stop_on, mode="min", min_delta=0.01))

    grad_clip = config['hparas'].get('gradient_clip_val', 1) if not config['hparas'].get('sep_class_opt', False) else False
    trainer = L.Trainer(
        logger=wandb_logger,
        precision="32",
        default_root_dir=args.exp_dir / config_path.stem,
        max_epochs=config['hparas']['epochs'],
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu", 
        strategy='ddp',
        # strategy=DDPStrategy(process_group_backend="gloo"),
        gradient_clip_val=grad_clip, 
        gradient_clip_algorithm=config['hparas'].get('gradient_clip_algorithm', "norm"),
        # limit_train_batches=2,
        # limit_val_batches=2,
        # val_check_interval=config['hparas']['valid_step'], # just validate every epoch 
        # check_val_every_n_epoch=5 if ("Matched" in config_path.stem) else 1, 
        profiler=None,
        plugins=[SLURMEnvironment(auto_requeue=False)], # so errors actually crash job 
        callbacks=callbacks)
    
    trainer.fit(model, ckpt_path=ckpt_path if args.resume_training else None)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='', type=str, help='Path to experiment config.')
    parser.add_argument('--config_list', type=str, help='Path to list of config files.')
    parser.add_argument('--array_id', type=int, help='Index into the config list specifying which one to use.')
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--ckpt_path",
        default='',
        type=str,
        help="Resume training from this checkpoint."
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=4,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 4)",
    )
    parser.add_argument(
    "--num_workers",
    default=0,
    type=int,
    help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument('--random_seed', default=0, type=int, help='Random seed for dataset.')
    parser.add_argument('--resume_training', action=argparse.BooleanOptionalAction, help='Resume training from checkpoint.')
    
    args = parser.parse_args()

    cli_main(args)
