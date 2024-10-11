import torch 
import lightning as L
import yaml
import sys, os
import pickle
from lightning_scripts.lightning_classifier import LitWordAudioSetModel 
from pathlib import Path 
import pathlib
from argparse import ArgumentParser

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def cli_main(args):
    L.seed_everything(args.random_seed)

    if args.config_path != "":
        config_path = pathlib.Path(args.config_path)
    elif args.config_list_path != "":
        with open(args.config_list_path, 'rb') as f:
            config_dict = pickle.load(f)
            config_path = pathlib.Path(config_dict[args.array_ix])

    print(f"Evaluating config: {config_path}")
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    config['num_workers'] = args.num_workers
    config['hparas']['batch_size'] = args.batch_size
    config['data']['eval_max'] = -1

    if args.ckpt_path == "":
        checkpoint_dir = Path(args.model_ckpt_dir) / f"{config_path.stem}/checkpoints"
        ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
        ckpt_path = ckpt_paths[-1] # get latest checkpoint 
        print(ckpt_path)
    else:
        ckpt_path = args.ckpt_path

    model = LitWordAudioSetModel.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)
    val_loader = model.val_dataloader() # will be populated with relevant args in config 
    trainer = L.Trainer(devices=args.gpus)
    trainer.validate(model, dataloaders=val_loader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='', type=str, help='Path to experiment config.')
    parser.add_argument('--config_list_path', default='', type=str, help='Path to experiment config.')
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
    parser.add_argument('--array_ix', default=0, type=int, help='Slurm job array index')
    args = parser.parse_args()

    cli_main(args)
