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
import robustness.audio_functions.audio_transforms as at 

from audio_ssl.misc import LARS, CosineWarmupScheduler
from jsinV3DataLoader_precombined_batched import jsinV3_precombined_all_signals
from torchmetrics.classification import Accuracy, BinaryPrecision
from robustness.audio_functions.jsinV3_loss_functions import jsinV3_multi_task_loss
import pathlib
from argparse import ArgumentParser, BooleanOptionalAction
from typing import List, Union, Tuple
import torchaudio 
sys.path.append('byol-a')
from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020
from easydict import EasyDict

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# TODO: refactor for different models - current setup is unreadable

class SSLClassifier(L.LightningModule):
    def __init__(self, config, ckpt_path, layer_out):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.layer_out = layer_out
        # init the pretrained LightningModule
        # Set strict to false to ignore loading in pre-trained classifier 
        self.feature_extractor = LitAudioSSL.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, strict=False).eval()
        self.feature_extractor = torch.compile(self.feature_extractor)
        self.feature_extractor.freeze()
        # softcode size dict at some point 
        self.time_avg_rep = config['model']['arch_kwargs'].get('time_average', True)

        if config['model']['arch_kwargs']['backbone'] == 'kell2018':
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
                layer_size_dict = {'avgpool': 512,
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
        
        
        proj_out_dim = layer_size_dict[layer_out]
        # init trainable word classifier  
        num_classes = config['model']['arch_kwargs']['num_classes']

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
        if isinstance(num_classes, dict): # Make multiple fully conected layers
            all_fc_layers = {}
            for task in num_classes.keys():
                all_fc_layers[task] = nn.Linear(proj_out_dim, num_classes[task]) 
            self.classifier = nn.ModuleDict(all_fc_layers)
        else:
            self.classifier = nn.Linear(proj_out_dim, num_classes)

        self.multi_task_loss = jsinV3_multi_task_loss(task_loss_params=config['hparas']['task_loss_params'],
                                                      batch_size=None,
                                                    #   reduction='none'
                                                    )

        self.accuracy = torch.nn.ModuleDict({task_key: BinaryPrecision() if 'noise' in task_key else Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_kwargs']['num_classes'].items()}) 
        
    def forward(self, x):
        with torch.no_grad():
            predictions, rep, all_outputs = self.feature_extractor.model(x,  with_latent=True, fake_relu=True)
            activations = all_outputs[self.layer_out]
            if self.time_avg_rep:
                activations = activations.mean(dim=-1).view(activations.shape[0], -1)
            else:
                activations = activations.view(activations.shape[0], -1)
                # time average then flatten
            activations = activations.detach()
        if self.mlp:
            activations = self.mlp(activations)
        if isinstance(self.classifier, nn.ModuleDict): 
            logits = {}
            for task, fc_l in self.classifier.items():
                logits[task] = fc_l(activations)
        else:
            logits = self.classifier(activations)
        return logits 

    def _step(self, batch, batch_idx, step_type):
        audio, labels = batch
        logits = self.forward(audio) 
        loss, task_loss_dict = self.multi_task_loss(logits, labels, return_indiv_loss=True)
        self.log(f"{step_type}_loss", loss.detach(), prog_bar=True, sync_dist = True)

        # calc acc, add acc and task loss to log
        for task, task_loss in task_loss_dict.items():
            task_acc = self.accuracy[task](logits[task], labels[task])
            # format task str for logging: remove 'noise/' or 'signal/' from str
            self.log(f"{step_type}_{task}_loss", task_loss.detach(),
                                             prog_bar=True, sync_dist = True)
            self.log(f"{step_type}_{task}_acc", task_acc, prog_bar=False, sync_dist = True)
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
        loss, task_loss_dict = self.multi_task_loss(logits, labels, return_indiv_loss=True)
        accuracy_dict = {}
        top5_dict = {}
        for task, task_loss in task_loss_dict.items():
            task_acc = self.accuracy[task](logits[task], labels[task])
            accuracy_dict[task] = task_acc
            task_top5 = torch.isin(torch.topk(logits[task].softmax(-1), k=5, dim=-1).indices, labels[task]).any(-1).float().mean()
            top5_dict[task] = task_top5
        return {"top1":accuracy_dict, "top5":top5_dict}

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
        batch = batch[0] # unbox wrapper added by dataloader 
        signals = []
        labels = batch[-1] # labels already collated 

        # convert labels to torch tensors 
        if isinstance(labels, dict):
            for task_key, task_labels in labels.items():
                labels[task_key] = torch.from_numpy(task_labels)
        else:
            labels = torch.from_numpy(labels) 
        # Only fit on clean targets 
        for (signal, noise) in  zip(*batch[:2]):
            # use transforms pre-defined in feature_extractor - None instead of noise to skip
            if self.config.get('with_noise', False):
                noise = noise
            else:
                noise = None 
            signal, _ = self.feature_extractor.transforms(signal, noise)
            if signal is None:
                # Signal was none & has null label class 
                signal = torch.zeros(1,40000)
            signals.append(signal)
        signals = torch.cat(signals).unsqueeze(1) # add back channel dim
        return signals, labels  

    def eval_collate_fn(self, batch):
        batch = batch[0] # unbox wrapper added by dataloader 
        signals = []
        labels = batch[-1] # labels already collated 

        # convert labels to torch tensors 
        if isinstance(labels, dict):
            for task_key, task_labels in labels.items():
                labels[task_key] = torch.from_numpy(task_labels)
        else:
            labels = torch.from_numpy(labels) 
        # Only fit on clean targets 
        for (signal, noise) in  zip(*batch[:2]):
            # use transforms pre-defined in feature_extractor - None instead of noise to skip
            signal, _ = self.feature_extractor.transforms(signal, None)
            if signal is None:
                # Signal was none & has null label class 
                signal = torch.zeros(1,40000)
            signals.append(signal)
        signals = torch.cat(signals).unsqueeze(1) # add back channel dim
        return signals, labels  

    
    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        dataset = jsinV3_precombined_all_signals(root="/mnt/ceph/users/jfeather/data/training_datasets_audio/JSIN_all_v3/subsets/",
                                                 train=True,
                                                 transform=None, # perform transforms in collate_fn
                                                 batch_size=self.config['hparas']['batch_size'])
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            # persistent_workers=True,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return train_dataloader
    
    def val_dataloader(self):
        dataset = jsinV3_precombined_all_signals(root="/mnt/ceph/users/jfeather/data/training_datasets_audio/JSIN_all_v3/subsets/",
                                                 train=False,
                                                 transform=None,
                                                 batch_size=self.config['hparas']['batch_size'],
                                                 eval_max=self.config['data'].get('eval_max', 3))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
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
    def __init__(self, config):
        super().__init__()
        config = EasyDict(config)
        self.config = config

        # self.stats = [10158236.,  51190964.]
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

        self.transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.CombineWithRandomDBSNR(low_snr=config['audio_transforms']['low_snr'],
                                        high_snr=config['audio_transforms']['high_snr']),
                at.RMSNormalizeForegroundAndBackground(rms_level=config['audio_transforms']['rms_level']),
                at.UnsqueezeAudio(dim=0) # dim=0 here so batches of audio from dataloader will be (Batch, 1, Time)
            ])       
        
        self.normalizer = PrecomputedNorm(self.stats)

        # Load pretrained weights.
        self.feature_extractor = AudioNTT2020(d=self.config.feature_d)
        self.feature_extractor.load_weight('byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', self.device)
        self.feature_extractor = self.feature_extractor.eval()
        # Need to manually freeze params here 
        self.feature_extractor.trainable = False
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False 

        num_classes = config['model']['arch_kwargs']['num_classes']
        proj_out_dim = 2048
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

        if isinstance(num_classes, dict): # Make multiple fully conected layers
            all_fc_layers = {}
            for task in num_classes.keys():
                all_fc_layers[task] = nn.Linear(proj_out_dim, num_classes[task]) 
            self.classifier = nn.ModuleDict(all_fc_layers)
        else:
            self.classifier = nn.Linear(proj_out_dim, num_classes)

        self.multi_task_loss = jsinV3_multi_task_loss(task_loss_params=config['hparas']['task_loss_params'],
                                                      batch_size=None,
                                                    #   reduction='none'
                                                    )
        self.accuracy = torch.nn.ModuleDict({task_key: BinaryPrecision() if 'noise' in task_key else Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_kwargs']['num_classes'].items()}) 
        
    def forward(self, x):
        with torch.no_grad():
            x = self.normalizer((self.to_melspec(x) + torch.finfo(torch.float).eps).log())
            activations = self.feature_extractor(x)
            activations = activations.detach()
        if self.mlp:
            activations = self.mlp(activations)
        if isinstance(self.classifier, nn.ModuleDict): 
            logits = {}
            for task, fc_l in self.classifier.items():
                logits[task] = fc_l(activations)
        else:
            logits = self.classifier(activations)
        return logits 

    def _step(self, batch, batch_idx, step_type):
        audio, labels = batch
        logits = self.forward(audio) 
        loss, task_loss_dict = self.multi_task_loss(logits, labels, return_indiv_loss=True)
        self.log(f"{step_type}_loss", loss.detach(), prog_bar=True, sync_dist = True)

        # calc acc, add acc and task loss to log
        for task, task_loss in task_loss_dict.items():
            task_acc = self.accuracy[task](logits[task], labels[task])
            # format task str for logging: remove 'noise/' or 'signal/' from str
            self.log(f"{step_type}_{task}_loss", task_loss.detach(),
                                             prog_bar=True, sync_dist = True)
            self.log(f"{step_type}_{task}_acc", task_acc, prog_bar=False, sync_dist = True)
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
        loss, task_loss_dict = self.multi_task_loss(logits, labels, return_indiv_loss=True)
        accuracy_dict = {}
        top5_dict = {}
        for task, task_loss in task_loss_dict.items():
            task_acc = self.accuracy[task](logits[task], labels[task])
            accuracy_dict[task] = task_acc
            task_top5 = torch.isin(torch.topk(logits[task].softmax(-1), k=5, dim=-1).indices, labels[task]).any(-1).float().mean()
            top5_dict[task] = task_top5
        return {"top1":accuracy_dict, "top5":top5_dict}

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
        batch = batch[0] # unbox wrapper added by dataloader 
        signals = []
        labels = batch[-1] # labels already collated 

        # convert labels to torch tensors 
        if isinstance(labels, dict):
            for task_key, task_labels in labels.items():
                labels[task_key] = torch.from_numpy(task_labels)
        else:
            labels = torch.from_numpy(labels) 
        # Only fit on clean targets 
        for (signal, noise) in  zip(*batch[:2]):
            # use transforms pre-defined in feature_extractor - None instead of noise to skip
            if self.config.get('with_noise', False):
                noise = noise
            else:
                noise = None 
            signal, _ = self.transforms(signal, noise)
            if signal is None:
                # Signal was none & has null label class 
                signal = torch.zeros(1,40000)
            signals.append(signal)
        signals = torch.cat(signals).unsqueeze(1) # add back channel dim
        return signals, labels  

    def eval_collate_fn(self, batch):
        batch = batch[0] # unbox wrapper added by dataloader 
        signals = []
        labels = batch[-1] # labels already collated 

        # convert labels to torch tensors 
        if isinstance(labels, dict):
            for task_key, task_labels in labels.items():
                labels[task_key] = torch.from_numpy(task_labels)
        else:
            labels = torch.from_numpy(labels) 
        # Only fit on clean targets 
        for (signal, noise) in  zip(*batch[:2]):
            # use transforms pre-defined in feature_extractor - None instead of noise to skip
            signal, _ = self.transforms(signal, None)
            if signal is None:
                # Signal was none & has null label class 
                signal = torch.zeros(1,40000)
            signals.append(signal)
        signals = torch.cat(signals).unsqueeze(1) # add back channel dim
        return signals, labels  
    
    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        dataset = jsinV3_precombined_all_signals(root="/mnt/ceph/users/jfeather/data/training_datasets_audio/JSIN_all_v3/subsets/",
                                                 train=True,
                                                 transform=None, # perform transforms in collate_fn
                                                 batch_size=self.config['hparas']['batch_size'])
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            # persistent_workers=True,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return train_dataloader
    
    def val_dataloader(self):
        dataset = jsinV3_precombined_all_signals(root="/mnt/ceph/users/jfeather/data/training_datasets_audio/JSIN_all_v3/subsets/",
                                                 train=False,
                                                 transform=None,
                                                 batch_size=self.config['hparas']['batch_size'],
                                                 eval_max=self.config['data'].get('eval_max', 3))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
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
    
    if args.config_path != "":
        config_path = pathlib.Path(args.config_path)
    elif args.config_list_path != "":
        with open(args.config_list_path, 'rb') as f:
            config_dict = pickle.load(f)
            config_path = pathlib.Path(config_dict[args.array_ix])

    print(f"Evaluating config: {config_path}")
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    use_byola = False 
    if 'byol-a' in str(config_path):
        config['model'] = {}
        config['hparas'] = {}
        config['hparas']['task_loss_params'] = {
            "signal/word_int":
                {"loss_type": 'crossentropyloss',
                "weight": 1.0},                       # init loss is ~6.6 
            "noise/labels_int":
                {"loss_type": 'bcewithlogitsloss',
                "weight": 1.0},                      # init loss is ~200 
            "signal/speaker_int":
                {"loss_type": 'crossentropyloss',
                "weight": 1.0}
            }
        config['audio_transforms'] = {} 
        config['audio_transforms']['low_snr'] = -10
        config['audio_transforms']['high_snr'] = 10
        config['audio_transforms']['rms_level'] = 60
        config['model']['arch_kwargs'] = {}
        config['data'] = {}
        use_byola = True 

    # update config for transfer learning task
    config['num_workers'] = args.num_workers
    config['num_gpus'] = args.gpus
    config['hparas']['batch_size'] = args.batch_size
    config['hparas']['global_batch_size'] = args.batch_size
    config['data']['eval_max'] = 3
    config['hparas']['optimizer'] = args.optimizer
    # used 2 gpus for training, mult by 2 for now to get same checkpoint 
    config['model']['arch_kwargs']['supervised'] =  False

    config['with_noise'] = args.with_noise
    # don't load in classifier head if it exists 
    config['hparas']['lr'] = args.lr 
    config['hparas']['epochs'] = 1
    config['model']['arch_kwargs']['time_average'] = args.time_avg_rep

    time_avg_str = ""
    if not args.time_avg_rep:
        time_avg_str = "full_rep_"

    scheduler_str = ""
    if args.lr_scheduler:
        config['hparas']['lr_schedule'] = True
        config['hparas']['num_warmup_steps_or_ratio'] = 0
        scheduler_str = "_cosine_lr_scheduler_"

    if args.task == 'both':
        config['model']['arch_kwargs']['num_classes'] = {"signal/word_int": 794,    
                                    "signal/speaker_int": 433} 
        task_str = f"word_and_speaker_task"
        
    elif args.task == 'word':
        config['model']['arch_kwargs']['num_classes'] = {"signal/word_int": 794} 
        task_str = f"word_task"

    elif args.task == 'speaker':
        config['model']['arch_kwargs']['num_classes'] = {"signal/speaker_int": 433} 
        task_str = f"speaker_task"

    print(f"Running {task_str} transfer")
    
    config['hparas']['task_loss_params'] = {key:value for key,value in config['hparas']['task_loss_params'].items() if key in config['model']['arch_kwargs']['num_classes'].keys()}

    if args.w_mlp:
        config['model']['classifier'] = {}
        config['model']['classifier']['hidden_dims'] = [args.mlp_dim]
        mlp_str = "_w_mlp"
    else:
        mlp_str = ""

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
    str_modifier = f"{task_str}_{args.layer_str}_{time_avg_str}{config['hparas']['optimizer']}_{config['hparas']['lr']}{scheduler_str}{mlp_str}{ckpt_modifier}{w_noise_modifier}"
    classifier_checkpoint_dir = pathlib.Path(args.model_ckpt_dir) / f"{config_path.stem}/linear_classifier_checkpoints_{str_modifier}"

    if use_byola:
        module = BYOLAClassifier(config=config)
    else:
        module = SSLClassifier(config=config,
                            ckpt_path=ckpt_path,
                            layer_out=args.layer_str)

    ## Check if existing classifier_ckpt exists 
    classifier_ckpts = list(classifier_checkpoint_dir.rglob("*.ckpt"))
    print(f"Existing classifier checkpoints: {classifier_ckpts}")
    classifier_ckpt = None 

    if len(classifier_ckpts) > 0 and args.use_classifier_ckpt:
        classifier_ckpt_path = sorted(classifier_ckpts, key=os.path.getctime)[-1]
        classifier_ckpt = torch.load(classifier_ckpt_path, weights_only=True) # get latest checkpoint 
        module.load_state_dict(classifier_ckpt['state_dict'])
        print(f"Loaded classifier from {classifier_ckpt_path}")
    
    if args.classifier_ckpt_path != '':
        classifier_ckpt_path = args.classifier_ckpt_path
        classifier_ckpt = torch.load(classifier_ckpt_path, weights_only=True) # get latest checkpoint 
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
        val_check_interval = 2000, 
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
    test_dataset = jsinV3_precombined_all_signals(root="/mnt/ceph/users/jfeather/data/training_datasets_audio/JSIN_all_v3/subsets/",
                                                train=False,
                                                transform=None,
                                                batch_size=config['hparas']['batch_size'],
                                                eval_max=-1)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config['num_workers'],
        shuffle=False,
        collate_fn=module.eval_collate_fn
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
    results_filename = results_dir / f"{config_path.stem}_linear_eval_jsin_{str_modifier}.pkl"
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
    parser.add_argument('--time_avg_rep', action=BooleanOptionalAction, help='Time average the model rep fed to classifer?')
    parser.add_argument('--use_classifier_ckpt', action=BooleanOptionalAction, help='Use existing classifer ckpt?')
    parser.add_argument('--mlp_dim', default=512, type=int, help='Hidden dim of MLP.')
    parser.add_argument('--lr_scheduler', action=BooleanOptionalAction, help='Use lr scheduler?')
    parser.add_argument('--array_ix', default=0, type=int, help='Slurm job array index')
    args = parser.parse_args()

    cli_main(args)
