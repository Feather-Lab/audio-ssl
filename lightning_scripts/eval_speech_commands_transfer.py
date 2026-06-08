import torch 
import torch.nn as nn 
import numpy as np 
import lightning as L
import yaml
import sys, os
import pickle
from lightning_ssl import LitAudioSSL
from lightning_classifier_matched_speech_in_noise import LitWordAudioSetModel as LitWordAudioSetModelMatched
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import robustness.audio_functions.audio_transforms as at 
from datasets import load_dataset

from audio_ssl.misc import LARS, CosineWarmupScheduler
from torchmetrics.classification import Accuracy
import pathlib
from torchaudio.functional import resample

from argparse import ArgumentParser, BooleanOptionalAction
from typing import List, Union, Tuple

import torchaudio 
sys.path.append('byol-a')
from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020
from easydict import EasyDict
from audiomae_speech_commands_module import AudioMAESpeechCommandsClassifier, WhisperSpeechCommandsClassifier

torch.set_float32_matmul_precision('medium')


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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class CenterCropOrPad:
    def __init__(self, sig_length):
        self.sig_length = sig_length

    def __call__(self, x):
        if x.shape[0] < self.sig_length:
        # edge pad if x is too short 
            pad_dur = (self.sig_length - len(x)) // 2 + 1 
            # print(f"X shape before pad: {x.shape}")
            x = nn.functional.pad(x, (pad_dur, pad_dur), "constant", 0 )
            # print(f"X shape after pad: {x.shape}")
        # re-compute crop bound

        # else:        
        start_idx = int((x.shape[0] - self.sig_length)/2)
        x = x[start_idx:start_idx+self.sig_length]

        return x


class SSLClassifier(L.LightningModule):
    def __init__(self, config, ckpt_path, layer_out, supervised_backbone=False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.layer_out = layer_out
        # init the pretrained LightningModule
        # Set strict to false to ignore loading in pre-trained classifier 
        if supervised_backbone:
            self.feature_extractor = LitWordAudioSetModelMatched.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, strict=False).eval()
            self.config['model']['arch_kwargs']['backbone'] = config['model']['arch_name']

        else:
            self.feature_extractor = LitAudioSSL.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, strict=False).eval()    
        self.feature_extractor = torch.compile(self.feature_extractor)
        self.feature_extractor.freeze()
        self.time_avg_rep = config['model']['arch_kwargs'].get('time_average', True)

        # softcode size dict at some point 

        if config['model']['arch_kwargs']['backbone'] == 'kell2018' or 'kell2018' in config['model']['arch_name']:
            if self.time_avg_rep:
                layer_size_dict = {'input_after_preproc': 211,
                                    'batchnorm0': 211,
                                    'conv0': 6816,
                                    'relu0': 6816,
                                    'maxpool0': 3456,
                                    'batchnorm1': 3456,
                                    'conv1': 4608,
                                    'relu1': 4608,
                                    'maxpool1': 2304,
                                    'batchnorm2': 2304,
                                    'conv2': 4608,
                                    'relu2': 4608,
                                    'conv3': 9216,
                                    'relu3': 9216,
                                    'conv4': 4608,
                                    'relu4': 4608,
                                    'avgpool': 2560,
                                    'xview': 23040,
                                    'fullyconnected': 4096,
                                    'relufc': 4096}
            else:
                layer_size_dict = {'input_after_preproc': 82290,
                                    'batchnorm0': 82290,
                                    'conv0': 886080,
                                    'relu0': 886080,
                                    'maxpool0': 224640,
                                    'batchnorm1': 224640,
                                    'conv1': 152064,
                                    'relu1': 152064,
                                    'maxpool1': 39168,
                                    'batchnorm2': 39168,
                                    'conv2': 78336,
                                    'relu2': 78336,
                                    'conv3': 156672,
                                    'relu3': 156672,
                                    'conv4': 78336,
                                    'relu4': 78336,
                                    'avgpool': 23040,
                                    'xview': 23040,
                                    'fullyconnected': 4096,
                                    'relufc': 4096}
            
        elif config['model']['arch_kwargs']['backbone'] == 'resnet18':
            if self.time_avg_rep:
                layer_size_dict = {'input_after_preproc': 211,
                                    'conv1': 6784,
                                    'bn1': 6784,
                                    'conv1_relu1': 6784,
                                    'maxpool1': 3392,
                                    'layer1': 3392,
                                    'layer2': 3456,
                                    'layer3': 3584,
                                    'layer4': 3584,
                                    'avgpool': 512,
                                    'final': 512}
            else:
                layer_size_dict = {'input_after_preproc': 82290,
                                    'conv1': 1322880,
                                    'bn1': 1322880,
                                    'conv1_relu1': 1322880,
                                    'maxpool1': 332416,
                                    'layer1': 332416,
                                    'layer2': 169344,
                                    'layer3': 89600,
                                    'layer4': 46592,
                                    'avgpool': 512,
                                    'final': 512}
        else:

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
        
        num_classes = 30 # 30 classes in speech commands dataset 

        proj_out_dim = layer_size_dict[layer_out]
        # init trainable word classifier  
        self.speech_commands_dataset = load_dataset("google/speech_commands", 'v0.01', trust_remote_code=True)

        self.crop_or_pad = CenterCropOrPad(40000)
        self.set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(60)

        self.mlp = None

        if config['model'].get('classifier', False):
            # Classifier is MLP defined by hparas
            # projection head (Following exactly barlow twins offical repo)
            hidden_dims = [proj_out_dim] + config['model']['classifier']['hidden_dims']
            layers = []
            for i in range(len(hidden_dims)-1):
                layers.append(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False)
                )
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                layers.append(nn.ReLU())
            proj_out_dim = hidden_dims[-1] 
            self.mlp = nn.Sequential(*layers)


        self.classifier = nn.Linear(proj_out_dim, num_classes)

        self.loss_fn = nn.CrossEntropyLoss() 

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes) 
        
    def forward(self, x):
        with torch.no_grad():
            predictions, rep, all_outputs = self.feature_extractor.model(x,  with_latent=True, fake_relu=False)
            activations = all_outputs[self.layer_out]
            if self.time_avg_rep:
                activations = activations.mean(dim=-1).view(activations.shape[0], -1)
            else:
                activations = activations.view(activations.shape[0], -1)
            activations = activations.detach()
        if self.mlp:
            activations = self.mlp(activations)
        logits = self.classifier(activations)
        return logits 

    def _step(self, batch, batch_idx, step_type):
        audio, labels = batch
        logits = self.forward(audio) 
        loss = self.loss_fn(logits,labels)

        self.log(f"{step_type}_loss", loss.detach(), prog_bar=True, sync_dist = True)
        # calc acc, add acc and task loss to log
        acc = self.accuracy(logits, labels)
        self.log(f"{step_type}_acc", acc, prog_bar=True, sync_dist = True)
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
        loss = self.loss_fn(logits,labels)
        task_acc = self.accuracy(logits, labels)
        task_top5 = torch.isin(torch.topk(logits.softmax(-1), k=5, dim=-1).indices, labels).any(-1).float().mean()
        return {"top1":task_acc, "top5": task_top5}

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
            lr = self.config['hparas']['lr']
            opt = getattr(torch.optim, self.config['hparas']['optimizer'])
            self.optimizer = opt(self.classifier.parameters(), lr=self.config['hparas']['lr']) 
                
        if self.config['hparas'].get('lr_schedule', False):
            total_training_steps = self.total_training_steps()
            num_warmup_steps = self.compute_warmup(total_training_steps, self.config['hparas']['num_warmup_steps_or_ratio'])
            lr_scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                batch_size=self.config['hparas']['global_batch_size'], # is global batch size
                warmup_steps=num_warmup_steps,
                max_steps=total_training_steps,
                lr=lr
            )
            return [self.optimizer], [
                    {
                        'scheduler': lr_scheduler,  # The LR scheduler instance (required)
                        'interval': 'step',  # The unit of the scheduler's step size
                    }
                ] 
        return [self.optimizer] # , [self.schedule]
    
    
    def collate_fn(self, batch):
        # batch = batch[0]
        tformed_audio = []
        word_labels = []
        for eg in batch:
            wav = torch.from_numpy(eg['audio']['array'])
            # wav, _ = audio_transforms(eg['array'], None)
            wav = resample(wav.float(), 16_000, 20_000)
            # zero pad word to middle of frame 
            wav = self.crop_or_pad(wav.squeeze())
            wav, _ = self.set_dbSPL(wav, None)
            if wav is None:
                continue
            tformed_audio.append(wav.unsqueeze(0))
            word_labels.append(eg['label'])
        audio = torch.stack(tformed_audio)
        word_int_label = torch.tensor(word_labels)
        return audio, word_int_label
    
    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        train_split = self.speech_commands_dataset['train']
        # remove silence 
        train_split = train_split.filter(lambda example: (not ('_silence_' in example['file'])) and (not (example['audio']['array'] is None)))
        train_dataloader = torch.utils.data.DataLoader(
            train_split,
            batch_size=self.config['hparas']['batch_size'],
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            # persistent_workers=True,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        return train_dataloader
    
    def val_dataloader(self):
        dataset = self.speech_commands_dataset['validation']
        # remove silence 
        dataset = dataset.filter(lambda example: (not ('_silence_' in example['file'])) and (not (example['audio']['array'] is None)))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['hparas']['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return dataloader
    
    def test_dataloader(self):
        dataset = self.speech_commands_dataset['test']
        # remove silence 
        dataset = dataset.filter(lambda example: (not ('_silence_' in example['file'])) and (not (example['audio']['array'] is None)))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['hparas']['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return dataloader

    def total_training_steps(self) -> int:
        dataset_size = len(self.train_dataloader())
        num_devices = self.config['num_gpus']
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps and self.trainer.max_steps != -1:
            return int(self.trainer.max_steps)
        return int(max_estimated_steps)

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_warmup_steps
    

class BYOLAClassifier(L.LightningModule):
    def __init__(self, config,):
        super().__init__()
        self.save_hyperparameters()
        super().__init__()
        config = EasyDict(config)
        self.config = config

        # self.stats = [10158236.,  51190964.] ## Stats of jsinV3 - can use if needeing to do inference
        self.stats = [-5.4919195,  5.0389895]

        self.to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        )
        
        self.normalizer = PrecomputedNorm(self.stats)

        # Load pretrained weights.
        self.feature_extractor = AudioNTT2020(d=self.config.feature_d)
        self.feature_extractor.load_weight('byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', self.device)
        self.feature_extractor = self.feature_extractor.eval()
        # Need to manually freeze params here 
        self.feature_extractor.trainable = False
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False 

        proj_out_dim = 2048

        num_classes = 30 # 30 classes in speech commands dataset 

        # init trainable word classifier  
        self.speech_commands_dataset = load_dataset("google/speech_commands", 'v0.01', trust_remote_code=True)

        self.crop_or_pad = CenterCropOrPad(40000)
        self.set_dbSPL = at.DBSPLNormalizeForegroundAndBackground(60)

        self.mlp = None

        if config['model'].get('classifier', False):
            # Classifier is MLP defined by hparas
            # projection head (Following exactly barlow twins offical repo)
            hidden_dims = [proj_out_dim] + config['model']['classifier']['hidden_dims']
            layers = []
            for i in range(len(hidden_dims)-1):
                layers.append(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False)
                )
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                layers.append(nn.ReLU())
            proj_out_dim = hidden_dims[-1] 
            self.mlp = nn.Sequential(*layers)


        self.classifier = nn.Linear(proj_out_dim, num_classes)

        self.loss_fn = nn.CrossEntropyLoss() 

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes) 
        
    def forward(self, x):
        with torch.no_grad():
            x = self.normalizer((self.to_melspec(x) + torch.finfo(torch.float).eps).log())
            activations = self.feature_extractor(x)
            activations = activations.detach()
        if self.mlp:
            activations = self.mlp(activations)
        logits = self.classifier(activations)
        return logits 

    def _step(self, batch, batch_idx, step_type):
        audio, labels = batch
        logits = self.forward(audio) 
        loss = self.loss_fn(logits,labels)

        self.log(f"{step_type}_loss", loss.detach(), prog_bar=True, sync_dist = True)
        # calc acc, add acc and task loss to log
        acc = self.accuracy(logits, labels)
        self.log(f"{step_type}_acc", acc, prog_bar=True, sync_dist = True)
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
        loss = self.loss_fn(logits,labels)
        task_acc = self.accuracy(logits, labels)
        task_top5 = torch.isin(torch.topk(logits.softmax(-1), k=5, dim=-1).indices, labels).any(-1).float().mean()
        return {"top1":task_acc, "top5": task_top5}

    def configure_optimizers(self):
        # Optimizer
        if self.config['hparas']['optimizer'] == "LARS":
            lr = self.config['hparas']['lr'] * self.config['hparas']['batch_size'] / 256
            self.optimizer = LARS(
                            self.classifier.parameters(),
                            lr=lr,
                            weight_decay=1e-6,
                            momentum=0.9,
                            weight_decay_filter=True,
                            lars_adaptation_filter=True,
                        ) 
        else:
            lr = self.config['hparas']['lr']
            opt = getattr(torch.optim, self.config['hparas']['optimizer'])
            self.optimizer = opt(self.classifier.parameters(), lr=self.config['hparas']['lr']) 
                
        if self.config['hparas'].get('lr_schedule', False):
            total_training_steps = self.total_training_steps()
            num_warmup_steps = self.compute_warmup(total_training_steps, self.config['hparas']['num_warmup_steps_or_ratio'])
            lr_scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                batch_size=self.config['hparas']['batch_size'], # is global batch size
                warmup_steps=num_warmup_steps,
                max_steps=total_training_steps,
                lr=lr
            )
            return [self.optimizer], [
                    {
                        'scheduler': lr_scheduler,  # The LR scheduler instance (required)
                        'interval': 'step',  # The unit of the scheduler's step size
                    }
                ] 
        return [self.optimizer] # , [self.schedule]
    
    
    def collate_fn(self, batch):
        # batch = batch[0]
        tformed_audio = []
        word_labels = []
        for eg in batch:
            wav = torch.from_numpy(eg['audio']['array'])
            # wav, _ = audio_transforms(eg['array'], None)
            wav = resample(wav.float(), 16_000, 20_000)
            # zero pad word to middle of frame 
            wav = self.crop_or_pad(wav.squeeze())
            wav, _ = self.set_dbSPL(wav, None)
            if wav is None:
                continue
            tformed_audio.append(wav.unsqueeze(0))
            word_labels.append(eg['label'])
        audio = torch.stack(tformed_audio)
        word_int_label = torch.tensor(word_labels)
        return audio, word_int_label
    
    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        train_split = self.speech_commands_dataset['train']
        # remove silence 
        train_split = train_split.filter(lambda example: (not ('_silence_' in example['file'])) and (not (example['audio']['array'] is None)))
        train_dataloader = torch.utils.data.DataLoader(
            train_split,
            batch_size=self.config['hparas']['batch_size'],
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            # persistent_workers=True,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        return train_dataloader
    
    def val_dataloader(self):
        dataset = self.speech_commands_dataset['validation']
        # remove silence 
        dataset = dataset.filter(lambda example: (not ('_silence_' in example['file'])) and (not (example['audio']['array'] is None)))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['hparas']['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return dataloader
    
    def test_dataloader(self):
        dataset = self.speech_commands_dataset['test']
        # remove silence 
        dataset = dataset.filter(lambda example: (not ('_silence_' in example['file'])) and (not (example['audio']['array'] is None)))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['hparas']['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return dataloader

    def total_training_steps(self) -> int:
        dataset_size = len(self.train_dataloader())
        num_devices = self.config['num_gpus']
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps and self.trainer.max_steps != -1:
            return int(self.trainer.max_steps)
        return int(max_estimated_steps)

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_warmup_steps
    


def cli_main(args):
    L.seed_everything(args.random_seed)

    use_audiomae = getattr(args, 'model_type', '') == 'audiomae'
    use_whisper_pretrained = getattr(args, 'model_type', '') == 'whisper'

    if use_audiomae:
        audiomae_layer_idx, audiomae_layer_name = _parse_audiomae_layer_str(args.layer_str)
        config_path = pathlib.Path('audiomae_pretrained')

        audiomae_config = {
            "encoder_layer": audiomae_layer_idx,
            "time_average": True,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "lr_schedule": bool(args.lr_scheduler),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_gpus": args.gpus,
            "classifier_hidden_dims": [args.mlp_dim] if args.w_mlp else None,
        }

        config = {
            'model': {'arch_kwargs': {}},
            'hparas': {
                'batch_size': args.batch_size,
                'optimizer': args.optimizer,
                'lr': args.lr,
                'epochs': 5,
            },
            'data': {'eval_max': 3},
            'num_workers': args.num_workers,
            'num_gpus': args.gpus,
        }
    elif use_whisper_pretrained:
        whisper_model_name = getattr(args, 'whisper_model', 'large-v3')
        import whisper as _whisper
        _tmp = _whisper.load_model(whisper_model_name)
        _n_blocks = len(_tmp.encoder.blocks)
        del _tmp
        from whisper_encoder_arch import whisper_layer_names_list
        layer_names = whisper_layer_names_list(_n_blocks)
        layer_str = args.layer_str
        if layer_str.isdigit():
            layer_str = f"encoder_block_{layer_str}"
        if layer_str not in layer_names:
            raise ValueError(f"Invalid whisper layer '{layer_str}'. Valid: {layer_names}")
        whisper_encoder_layer = int(layer_str.split("_")[-1]) if layer_str.startswith("encoder_block_") else _n_blocks
        whisper_layer_name = layer_str

        config_path = pathlib.Path(f'whisper_{whisper_model_name}')
        whisper_config = {
            "whisper_model": whisper_model_name,
            "encoder_layer": whisper_encoder_layer,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "lr_schedule": bool(args.lr_scheduler),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_gpus": args.gpus,
            "classifier_hidden_dims": [args.mlp_dim] if args.w_mlp else None,
        }

        config = {
            'model': {'arch_kwargs': {}},
            'hparas': {
                'batch_size': args.batch_size,
                'optimizer': args.optimizer,
                'lr': args.lr,
                'epochs': 5,
            },
            'data': {'eval_max': 3},
            'num_workers': args.num_workers,
            'num_gpus': args.gpus,
        }
    else:
        if args.config_path != "":
            config_path = pathlib.Path(args.config_path)
        elif args.config_list_path != "":
            with open(args.config_list_path, 'rb') as f:
                config_dict = pickle.load(f)
                config_path = pathlib.Path(config_dict[args.array_ix])
        else:
            raise ValueError("Must provide either config_path, config_list_path, or --model_type audiomae/whisper")

        print(f"Evaluating config: {config_path}")
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    use_byola = False
    if not use_audiomae and 'byol-a' in str(config_path):
        use_byola = True 
        config['model'] = {}
        config['hparas'] = {}
        config['audio_transforms'] = {} 
        config['model']['arch_kwargs'] = {}
        config['data'] = {}

    config['num_workers'] = args.num_workers
    config['num_gpus'] = args.gpus
    config['hparas']['batch_size'] = args.batch_size
    config['data']['eval_max'] = 3
    config['hparas']['optimizer'] = args.optimizer
    config['hparas']['lr'] = args.lr 
    config['hparas']['epochs'] = args.train_epochs

    if not use_audiomae:
        if not args.supervised_backbone:
            config['model']['arch_kwargs']['supervised'] = False
        if 'arch_kwargs' not in config['model'].keys():
            config['model']['arch_kwargs'] = {}
        config['model']['arch_kwargs']['time_average'] = args.time_avg_rep

    time_avg_str = ""
    if not use_audiomae and not args.time_avg_rep:
        time_avg_str = "full_rep_"

    scheduler_str = ""
    if args.lr_scheduler:
        if not use_audiomae:
            config['hparas']['lr_schedule'] = True
            config['hparas']['num_warmup_steps_or_ratio'] = 0
        scheduler_str = "_cosine_lr_scheduler_"

    print(f"Running speech commands transfer")
    
    if args.w_mlp:
        if not use_audiomae:
            config['model']['classifier'] = {}
            config['model']['classifier']['hidden_dims'] = [args.mlp_dim]
        mlp_str = "_w_mlp"
    else:
        mlp_str = ""

    ckpt_path = None
    ckpt_modifier = ''
    if use_audiomae or use_whisper_pretrained:
        checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/checkpoints"
    else:
        checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/checkpoints"
        if args.ckpt_path == "":
            ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
            if len(ckpt_paths) > 0:
                ckpt_path = ckpt_paths[-1]
                print(ckpt_path)
        else:
            ckpt_path = args.ckpt_path
            ckpt_modifier = '_from_best_val_ckpt'

    if use_audiomae:
        layer_component = audiomae_layer_name
    elif use_whisper_pretrained:
        layer_component = whisper_layer_name
    else:
        layer_component = args.layer_str
    str_modifier = f"{config['hparas']['optimizer']}_{layer_component}_{time_avg_str}{config['hparas']['lr']}{scheduler_str}{mlp_str}{ckpt_modifier}"
    classifier_checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/speech_commands_linear_classifier_checkpoints/{str_modifier}"

    if use_audiomae:
        module = AudioMAESpeechCommandsClassifier(config=audiomae_config)
    elif use_whisper_pretrained:
        module = WhisperSpeechCommandsClassifier(config=whisper_config)
    elif use_byola:
        module = BYOLAClassifier(config=config)
    else:
        module = SSLClassifier(config=config,
                           ckpt_path=ckpt_path,
                           layer_out=args.layer_str,
                            supervised_backbone=args.supervised_backbone)

    ## Check if existing classifier_ckpt exists 
    classifier_ckpts = list(classifier_checkpoint_dir.rglob("*.ckpt"))
    print(f"Existing classifier checkpoints: {classifier_ckpts}")
    classifier_ckpt = None 

    if len(classifier_ckpts) > 0 and args.classifier_ckpt_path == '':
        classifier_ckpt_path = sorted(classifier_ckpts, key=os.path.getctime)[-1]
        classifier_ckpt = torch.load(classifier_ckpt_path, weights_only=False) # get latest checkpoint 
        module.load_state_dict(classifier_ckpt['state_dict'])
        print(f"Loaded classifier from {classifier_ckpt_path}")
    
    if args.classifier_ckpt_path != '':
        classifier_ckpt_path = args.classifier_ckpt_path
        classifier_ckpt = torch.load(classifier_ckpt_path, weights_only=False) # get latest checkpoint 
        module.load_state_dict(classifier_ckpt['state_dict'])
        print(f"Loaded classifier from {classifier_ckpt_path}")
        

    callbacks=[]
    callbacks.append(ModelCheckpoint(
            classifier_checkpoint_dir,
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ))
    # callbacks.append(EarlyStopping(monitor="train_classifier_loss", mode="min"))
    log_basename = 'byol-a_base' if use_byola else config_path.stem

    wandb_logger = WandbLogger(save_dir=checkpoint_dir, 
                               name=f"{log_basename}_speech_commands_classifier_{str_modifier}",
                               group='speech_commands_classifier',
                               project='cochdnn')

    trainer = L.Trainer(
        precision="32",
        # limit_val_batches=0,
        default_root_dir=args.model_ckpt_dir / config_path.stem,
        max_epochs=config['hparas']['epochs'],
        devices=args.gpus,
        accelerator="gpu", 
        strategy='ddp' if args.gpus > 1 else 'auto',
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

    # run test 

    print("Running inference")
    test_dataloader = module.test_dataloader()
    outputs = trainer.predict(module, test_dataloader, return_predictions=True)
    # get stats from test 
    top1_word = []
    top5_word = []

    for record in outputs:
        top1_word.append(record['top1'])
        top5_word.append(record['top5'])
    n_examples = len(outputs)
    

    output_dict = {
        "word_top1_mean": torch.stack(top1_word).mean(),
        "word_top1_sem": torch.stack(top1_word).std() / np.sqrt(n_examples),
        "word_top5_mean": torch.stack(top5_word).mean(),
        "word_top5_sem": torch.stack(top5_word).std() / np.sqrt(n_examples),
    }
        
    output_dict = {key:val.item() for key,val in output_dict.items()}  

    print(output_dict)
    # save results as .pkl 
    results_dir = pathlib.Path(args.results_dir) / "linear_eval_speech_commands"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_filename = results_dir / f"{config_path.stem}_{str_modifier}.pkl"
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
    parser.add_argument('--layer_str', default='avgpool', type=str, help='Layer to fit classifier ontop of.')
    parser.add_argument('--task', default='both', type=str, help='One of: ["both", "word", "speaker"]. Default is "both"')
    parser.add_argument('--optimizer', default='LARS', type=str, help='String for optimizer used.')
    parser.add_argument('--lr', default=0.2, type=float, help='Initial LR used.')
    parser.add_argument('--w_mlp', action=BooleanOptionalAction, help='Use MLP instead of linear classifier?')
    parser.add_argument('--with_noise', action=BooleanOptionalAction, help='Include noise in training?')
    parser.add_argument('--overwrite_classifier', action=BooleanOptionalAction, help='Overwrite existing classifer?')
    parser.add_argument('--eval_only', action=BooleanOptionalAction, help='Eval using existing classifer?')
    parser.add_argument('--mlp_dim', default=512, type=int, help='Hidden dim of MLP.')
    parser.add_argument('--lr_scheduler', action=BooleanOptionalAction, help='Use lr scheduler?')
    parser.add_argument('--array_ix', default=0, type=int, help='Slurm job array index')
    parser.add_argument('--time_avg_rep', action=BooleanOptionalAction, help='Time average the model rep fed to classifer?')
    parser.add_argument('--supervised_backbone', action=BooleanOptionalAction, help='Using supervised backbone?')
    parser.add_argument('--train_epochs', default=5, type=int, help='Number of training epochs.')
    parser.add_argument('--model_type', default='', type=str, help='Model type: "audiomae" or "whisper" for pretrained encoders (no config file needed).')
    parser.add_argument('--whisper_model', default='large-v3', type=str, help='Whisper model name for --model_type whisper (e.g. large-v3).')

    args = parser.parse_args()

    cli_main(args)
