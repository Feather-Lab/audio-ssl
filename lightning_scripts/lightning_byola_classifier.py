import torch 
import torch.nn as nn 
import numpy as np 
import lightning as L
from typing import Union
import torchaudio 
import sys
sys.path.append('byol-a')
from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020
from easydict import EasyDict
from audio_ssl.misc import LARS, CosineWarmupScheduler
from jsinV3DataLoader_precombined_batched import jsinV3_precombined_all_signals
from torchmetrics.classification import Accuracy, BinaryPrecision
from robustness.audio_functions.jsinV3_loss_functions import jsinV3_multi_task_loss
import robustness.audio_functions.audio_transforms as at 

class BYOLAClassifier(L.LightningModule):
    def __init__(self, config):
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

        # hack so resample not put on gpu 
        self.resample_audio = lambda x:  torchaudio.functional.resample(x, orig_freq=20_000, new_freq=16_000)

        self.transforms = at.AudioCompose([
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
        print(self.feature_extractor)
        # Need to manually freeze params here 
        self.feature_extractor.trainable = False
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False 

        # Layer selection for classifier attachment
        self.layer_name = config.get('classifier_layer', None)  # e.g., 'layer3', 'avgpool', None for final
        self.layer_output = None
        
        if self.layer_name:
            self._register_hook()
            proj_out_dim = self._get_layer_output_dim()
        else:
            proj_out_dim = 2048  # Default final output dimension

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
    

    def _register_hook(self):
        """Register forward hook to capture intermediate layer outputs"""
        def hook_fn(module, input, output):
            self.layer_output = output
        
        # Parse layer specification (e.g., 'features.2' or 'features.6')
        parts = self.layer_name.split('.')
        
        # Navigate to the target module
        target_module = self.feature_extractor
        for part in parts:
            if part.isdigit():
                # Access Sequential layer by index
                target_module = target_module[int(part)]
            else:
                # Access by attribute name
                target_module = getattr(target_module, part)
        
        target_module.register_forward_hook(hook_fn)

    def _get_layer_output_dim(self):
        """Probe the layer to get its output dimension"""
        with torch.no_grad():
             # Typical mel-spectrogram shape for BYOLA for 2 second audio at 16kHz
            dummy_input = torch.randn(1, 1, 64, 201).to(self.device) 
            _ = self.feature_extractor(dummy_input)
            
            if self.layer_output is None:
                raise ValueError(f"Layer '{self.layer_name}' did not produce output. Check layer name.")
            
            # Handle different output shapes
            output_dim = self.layer_output.flatten(start_dim=1).shape[-1]
            self.layer_output = None  # Reset
            return output_dim
        
    def forward(self, x):
        with torch.no_grad():
            x = self.normalizer((self.to_melspec(x) + torch.finfo(torch.float).eps).log())
            if self.layer_name:
                # Run through feature extractor to populate layer_output via hook
                _ = self.feature_extractor(x)
                activations = self.layer_output.detach()
                self.layer_output = None  # Reset for next forward pass
                
                # Flatten if needed
                if len(activations.shape) > 2:
                    activations = activations.reshape(activations.shape[0], -1)
            else:
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
            # filter examples with null label
            task_IXS = (labels[task] != 0 ).nonzero(as_tuple=True)
            task_logits = logits[task][task_IXS]
            task_labels =  labels[task][task_IXS]
            task_acc = self.accuracy[task](task_logits, task_labels)
            accuracy_dict[task] = task_acc
            task_top5 = torch.isin(torch.topk(task_logits.softmax(-1), k=5, dim=-1).indices, task_labels).any(-1).float().mean()
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
                noise = self.resample_audio(torch.from_numpy(noise))
            else:
                noise = None 
            signal = self.resample_audio(torch.from_numpy(signal))
            signal, _ = self.transforms(signal, noise)
            if signal is None:
                # Signal was none & has null label class 
                signal = torch.zeros(1,32000)
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
            signal = self.resample_audio(torch.from_numpy(signal))
            signal, _ = self.transforms(signal, None)
            if signal is None:
                # Signal was none & has null label class 
                signal = torch.zeros(1,32000)
            signals.append(signal)
        signals = torch.cat(signals).unsqueeze(1) # add back channel dim
        return signals, labels  

    def predict_collate_fn(self, batch):
        audio, targets = batch[0] # unbox wrapper added by dataloader 
        # audio = audio.unsqueeze(1)
        signals = []
        for signal in audio:
            signal = self.resample_audio(signal)
            signal, _ = self.transforms(signal, None)
            if signal is None:
                # Signal was none & has null label class 
                signal = torch.zeros(1,32000)
            signals.append(signal)   
        signals = torch.cat(signals).unsqueeze(1) # add back channel dim

        # # combine labels: each target is dict for each key, stack the values 
        labels = {}
        for label_key in targets.keys():
            labels[label_key] = torch.from_numpy(targets[label_key])
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