import torch 
import torch.nn as nn 
import numpy as np 
import lightning as L
import yaml
import sys, os
import pickle
from lightning_ssl import LitAudioSSL 
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from audio_ssl.misc import LARS
from jsinV3DataLoader_precombined_batched import jsinV3_precombined_all_signals
from metrics import calculate_accuracy
import pathlib
from argparse import ArgumentParser

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class SSLWordClassifier(L.LightningModule):
    def __init__(self, config, ckpt_path, layer_out):
        super().__init__()
        self.config = config
        self.layer_out = layer_out
        # init the pretrained LightningModule
        self.feature_extractor = LitAudioSSL.load_from_checkpoint(checkpoint_path=ckpt_path, config=config).eval()
        self.feature_extractor = torch.compile(self.feature_extractor)

        self.feature_extractor.freeze()
        # softcode size dict at some point 
        layer_size_dict = {'input_after_preproc': 211,
                            'conv1': 6784,
                            'bn1': 6784,
                            'conv1_relu1': 6784,
                            'maxpool1': 3392,
                            'layer1': 13568,
                            'layer2': 13824,
                            'layer3': 14336,
                            'layer4': 14336,
                            'avgpool': 2048,
                            'final': 2048}
        
        proj_out_dim = layer_size_dict[layer_out]
        # init trainable word classifier  
        self.classifier = torch.nn.Linear(proj_out_dim,
                                          config['model']['arch_kwargs']['n_classes'])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        with torch.no_grad():
            predictions, rep, all_outputs = self.feature_extractor.model(x,  with_latent=True, fake_relu=True)
            activations = all_outputs[self.layer_out]

            if self.layer_out == 'avgpool' or self.layer_out == 'final':
                activations = activations.view(self.config['hparas']['batch_size'], -1)
            else:
                # time average then flatten
                activations = activations.mean(dim=-1).view(self.config['hparas']['batch_size'], -1)
        x = self.classifier(activations)
        return x 

    def _step(self, batch, batch_idx, step_type):
        audio, labels = batch
        logits = self.forward(audio) 
        loss = self.loss(logits, labels)
        self.log(f"{step_type}_loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
        return loss 
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def predict_step(self, batch):
        audio, labels = batch
        logits = self.forward(audio) 
        loss = self.loss(logits, labels)
        accuracy = calculate_accuracy(logits.softmax(-1), labels, reduce=False)
        # self.log(f"test_loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
        # self.log(f"test_accuray", accuracy.detach(), on_step=True, on_epoch=False, prog_bar=True)
        return {'accuracy':accuracy} 


    def configure_optimizers(self):
        # Optimizer
        if self.config['hparas']['optimizer'] == "LARS":
            lr = self.config['hparas']['lr'] * self.config['hparas']['global_batch_size'] / 256
            self.optimizer = LARS(
                            self.classifier.parameters(),
                            lr=lr,
                            weight_decay=1e-6,
                            momentum=0.9,
                            weight_decay_filter=True,
                            lars_adaptation_filter=True,
                        ) 
        else:
            opt = getattr(torch.optim, self.config['hparas']['optimizer'])
            self.optimizer = opt(self.classifier.parameters(), lr=self.config['hparas']['lr'])     
            # self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['hparas']['step_lr']) 
        return [self.optimizer] # , [self.schedule]
    
    def collate_fn(self, batch):
        batch = batch[0] # unbox wrapper added by dataloader 
        signals = []
        labels = batch[-1] # labels already collated 
        # convert labels to torch tensors 
        if isinstance(labels, dict):
            # hardcode selection of word labels 
            labels = torch.from_numpy(labels['signal/word_int'])
        else:
            labels = torch.from_numpy(labels) 
        # convert signal and noise into signal
        for (signal, noise) in  zip(*batch[:2]):
            # use transforms pre-defined in feature_extractor
            signal, _ = self.feature_extractor.transforms(signal, noise)
            signals.append(signal)
        signals = torch.cat(signals).unsqueeze(1) # add back channel dim
        return signals, labels  
    
    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        dataset = jsinV3_precombined_all_signals(root=self.config['data']['root'],
                                                 train=True,
                                                 transform=None, # perform transforms in collate_fn
                                                 batch_size=self.config['hparas']['batch_size'])
        dataset.target_keys = ['signal/word_int']
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            # persistent_workers=True,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return self.train_dataloader
    
    def val_dataloader(self):
        dataset = jsinV3_precombined_all_signals(root=self.config['data']['root'],
                                                 train=False,
                                                 transform=None,
                                                 batch_size=self.config['hparas']['batch_size'],
                                                 eval_max=self.config['data'].get('eval_max', 3))
        dataset.target_keys = ['signal/word_int']
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return dataloader


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
    config['data']['eval_max'] = 3
    config['hparas']['optimizer'] = "LARS"
    config['hparas']['lr'] = 0.6
    config['hparas']['epochs'] = 10


    # get checkpoint for ssl model 
    if args.ckpt_path == "":
        checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/checkpoints"
        ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
        ckpt_path = ckpt_paths[-1] # get latest checkpoint 
        print(ckpt_path)
    else:
        ckpt_path = args.ckpt_path

    classifier_checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/linear_classifier_checkpoints"

    module = SSLWordClassifier(config=config,
                           ckpt_path=ckpt_path,
                           layer_out=args.layer_str)
    callbacks=[]
    callbacks.append(ModelCheckpoint(
            classifier_checkpoint_dir,
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ))
    callbacks.append(EarlyStopping(monitor="train_loss", mode="min"))

    wandb_logger = WandbLogger(save_dir=checkpoint_dir, 
                               name=f"{config_path.stem}_word_classifier",
                               group='word_classifier_transfer',
                               project='cochdnn')

    trainer = L.Trainer(
        precision="32",
        # limit_val_batches=0,
        default_root_dir=args.model_ckpt_dir / config_path.stem,
        max_epochs=config['hparas']['epochs'],
        devices=args.gpus,
        accelerator="gpu", 
        gradient_clip_val=1, # clipt grad l2 norm to 1 
        profiler=None,
        logger=wandb_logger,
        callbacks=callbacks)   
    
    # fit classifier 
    trainer.fit(module)

    # run test 
    test_dataset = jsinV3_precombined_all_signals(root=config['data']['root'],
                                                 train=False,
                                                 transform=None,
                                                 batch_size=config['hparas']['batch_size'],
                                                 eval_max=-1)
    test_dataset.target_keys = ['signal/word_int']
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=self.config['num_workers'],
        shuffle=False,
        collate_fn=self.collate_fn
    )

    outputs = trainer.predict(module, test_dataloader, return_predictions=True)
    # get stats from test 
    output_vals = torch.cat([output['accuracy'] for output in outputs])
    n_examples = output_vals.shape[0]
    output_dict = {
            "mean_acc": output_vals.mean(),
            "std_acc" :output_vals.std(),
            "sem_acc": output_vals.std() / np.sqrt(n_examples)
        }   
    # save results as .pkl 
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_filename = results_dir / f"{config_path.stem}_linear_eval_jsin.pkl"
    with open(results_filename, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                       

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='', type=str, help='Path to experiment config.')
    parser.add_argument('--config_list_path', default='', type=str, help='Path to experiment config.')
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
    parser.add_argument('--layer_str', default='avgpool', type=str, help='Layer to fit classifier ontop of.')
    parser.add_argument('--array_ix', default=0, type=int, help='Slurm job array index')
    args = parser.parse_args()

    cli_main(args)
