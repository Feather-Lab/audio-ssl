
import torch
from torch import nn
from torchmetrics.classification import Accuracy, BinaryPrecision
import torch.nn.functional as F
import lightning as L
from typing import List, Union, Tuple

import sys

import robustness.audio_models as architectures
import robustness.audio_functions.audio_transforms as at 
from robustness.audio_functions.jsinV3_loss_functions import jsinV3_multi_task_loss
from robustness.audio_functions.audio_input_representations import AUDIO_INPUT_REPRESENTATIONS

sys.path.append('./lightning_scripts/')
from audio_ssl.misc import LARS, CosineWarmupScheduler
from jsinV3DataLoader_precombined_batched import (
    MatchedSpeechInNoiseDatasetBatched,
    MatchedAudiosetBatched,
)



class ModelWithFrontEnd(nn.Module):
    def __init__(self,front_end, model):
        super().__init__()
        self.front_end = front_end
        self.model = model

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        x, _ = self.front_end(x, None)
        return self.model(x,  with_latent=with_latent, fake_relu=fake_relu, no_relu=no_relu)

class LitWordAudioSetModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 

        # Get audio config and init representation 
        self.audio_config = AUDIO_INPUT_REPRESENTATIONS[config['audio_rep']['name']]
        self.audio_rep = at.AudioToAudioRepresentation(**self.audio_config)

        # Get audio model from config kwargs
        self.model = architectures.__dict__[self.config['model']['arch_name']](**self.config['model']['arch_params'])
        if 'resnet' in self.config['model']['arch_name']:
            self.metamer_layers = [
                'input_after_preproc',
                'conv1',
                'bn1',
                'conv1_relu1',
                'maxpool1',
                'layer1',
                'layer2',
                'layer3',
                'layer4',
                'avgpool',
            ]
            
        elif 'kell2018' in self.config['model']['arch_name']:
            self.metamer_layers = [
                'input_after_preproc',
                'batchnorm0',
                'conv0',
                'relu0',
                'maxpool0',
                'batchnorm1',
                'conv1',
                'relu1',
                'maxpool1',
                'batchnorm2',
                'conv2',
                'relu2',
                'conv3',
                'relu3',
                'conv4',
                'relu4',
                'avgpool',
                'fullyconnected',
                'relufc',
                'dropout',
                ]

        self.model = ModelWithFrontEnd(self.audio_rep, self.model)

        # Track dataset type (MatchedSpeechInNoiseDatasetBatched default)
        self.dataset_name = self.config['data'].get('dataset', "MatchedSpeechInNoiseDatasetBatched")
        self.dataset_common_kwargs = {
            'low_db': self.config['audio_transforms']['low_snr'],
            'high_db': self.config['audio_transforms']['high_snr'],
            'db_spl': self.config['audio_transforms']['dbspl'],
            'batch_size': self.config['hparas']['batch_size'],
            'target_keys': self.config['data']['target_keys'],
            'blocked_batches': self.config['data'].get('blocked_batches', True),
            'signal_augment': self.config['data'].get('signal_augment', False),
            'skip_aug_match': self.config['data'].get('skip_aug_match', False),
            'clean_percentage': self.config['data'].get('clean_percentage', 0.0),
            'overfit': self.config['data'].get('overfit', False),
        }

    
        self.multi_task_loss = jsinV3_multi_task_loss(task_loss_params=config['hparas']['task_loss_params'],
                                                      batch_size=None,
                                                    #   reduction='none'
                                                    )

        # get accuracy metrics per task - requires module dict for torchmetrics 
        self.train_accuracy = torch.nn.ModuleDict({task_key: BinaryPrecision() if 'noise' in task_key else Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_params']['num_classes'].items()}) 
        
        self.val_accuracy = torch.nn.ModuleDict({task_key: BinaryPrecision() if 'noise' in task_key else Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_params']['num_classes'].items()}) 
        
        self.accuracy = {'train': self.train_accuracy, 'val': self.val_accuracy}

    def _step(self, batch, batch_idx, step_type):
        audio, label_dict = batch
        logits = self.model(audio)

        # get classification loss
        loss, task_loss_dict = self.multi_task_loss(logits, label_dict, return_indiv_loss=True)
        self.log(f"{step_type}_loss", loss.detach(), prog_bar=True, sync_dist = True)
        
        # calc acc, add acc and task loss to log
        for task, task_loss in task_loss_dict.items():
            task_acc = self.accuracy[step_type][task](logits[task], label_dict[task])
            # format task str for logging: remove 'noise/' or 'signal/' from str
            self.log(f"{step_type}_{task}_loss", task_loss.detach(),
                                            #  on_step=True, on_epoch=False,
                                             prog_bar=True, sync_dist = True)
            self.log(f"{step_type}_{task}_acc", task_acc,
                                                        #  on_step=True,
                                                        #   on_epoch=False,
                                                          prog_bar=False, sync_dist = True)
        # log current learning rate 
        # lr = self.schedule.get_last_lr()[0]
        # self.log(f"lr", lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
    
    def on_before_optimizer_step(self, _):
        def _get_grad_norm(params, scale=1):
            """Compute grad norm given a gradient scale."""
            total_norm = 0.0
            for p in params:
                if p.grad is not None:
                    param_norm = (p.grad.detach().data / scale).norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            return total_norm
        grad_norm = _get_grad_norm(self.model.parameters())
        self.log("grad_norm", torch.tensor(grad_norm), prog_bar=True, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        # Optimizer
        if self.config['hparas']['optimizer'] == "LARS":
            # Typical learning rate scheduling is handled in CosineWarmupScheduler
            # as init_lr * batchsize / 256
            # lr given to LARS is 0
            #  CosineWarmupScheduler handles incrementing the LR
            self.optimizer = LARS(
                            self.model.parameters(),
                            lr=0,
                            weight_decay=1e-6,
                            momentum=0.9,
                            weight_decay_filter=True,
                            lars_adaptation_filter=True,
                        )
            total_training_steps = self.total_training_steps()
            num_warmup_steps = self.compute_warmup(total_training_steps, self.config['hparas']['num_warmup_steps_or_ratio'])
            lr_scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                batch_size=self.config['hparas']['global_batch_size'], # is global batch size
                warmup_steps=num_warmup_steps,
                max_steps=total_training_steps,
                lr=self.config['hparas']['lr']
            )
            return [self.optimizer], [
                {
                    'scheduler': lr_scheduler,  # The LR scheduler instance (required)
                    'interval': 'step',  # The unit of the scheduler's step size
                }
            ]    

        else:
            opt = getattr(torch.optim, self.config['hparas']['optimizer'])
            self.optimizer = opt(self.model.parameters(),  **self.config['hparas']['optimizer_kwargs'])     
            if self.config['hparas'].get('step_lr', False):
                self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['hparas']['step_lr']) 
                return [self.optimizer], [
                    {
                        'scheduler': self.schedule,  # The LR scheduler instance (required)
                        'interval': 'epoch',  # The unit of the scheduler's step size
                    }
                ]
            else:
                total_training_steps = self.total_training_steps()
                num_warmup_steps = self.compute_warmup(total_training_steps, self.config['hparas']['num_warmup_steps_or_ratio'])
                lr_scheduler = CosineWarmupScheduler(
                    optimizer=self.optimizer,
                    batch_size=self.config['hparas']['global_batch_size'], # is global batch size
                    warmup_steps=num_warmup_steps,
                    max_steps=total_training_steps,
                    lr=self.config['hparas']['optimizer_kwargs']['lr']
                )
                return [self.optimizer],   [
                        {
                                
                            'scheduler': lr_scheduler,  # The LR scheduler instance (required)
                            'interval': 'step',  # The unit of the scheduler's step size
                    }
                ]
    def forward(self, x):
        """
        PL required forward wrapper. Enables calling model in two ways:
        1) standard call in .py scripts
            model = LitAudioSSL(args)
            outs = model(inputs)
        2) inside this lightning module's methods as self (eg in _step)
            outs = self(inputs) # self is self.forward, and is same as self.model.forward 
        """
        return self.model(x)

    def collate_fn(self, batch):
        audio, targets = batch[0] # is [comb11, comb12, comb21, comb22], [targ11, targ12, targ21, targ22]
        audio = torch.vstack(audio).unsqueeze(1)
        # # combine labels: each target is dict for each key, stack the values 
        labels = {}
        for label_key in targets[0].keys():
            stacked = torch.concat([label_set[label_key] for label_set in targets])
            if stacked.ndim > 1 and stacked.shape[-1] < 527:
                pad_width = 527 - stacked.shape[-1]
                stacked = F.pad(stacked, (0, pad_width), mode='constant', value=0)
            labels[label_key] = stacked
        return audio, labels

    def train_dataloader(self):
        if self.dataset_name == "MatchedAudiosetBatched":
            dataset = MatchedAudiosetBatched(
                noise_h5_path=self.config['data']['noise_h5_path'],
                low_db=self.config['audio_transforms']['low_snr'],
                high_db=self.config['audio_transforms']['high_snr'],
                db_spl=self.config['audio_transforms']['dbspl'],
                batch_size=self.config['hparas']['batch_size'],
                target_keys=self.config['data'].get("target_keys", None),
                blocked_batches=self.config['data'].get("blocked_batches", False),
                signal_augment=self.config['data'].get("signal_augment", False),
                skip_aug_match=self.config['data'].get("skip_aug_match", False),
                clean_percentage=self.config['data'].get("clean_percentage", 0.0),
                in_sample_rate=self.config['data'].get('in_sample_rate', 16_000),
                out_sample_rate=self.config['data'].get('out_sample_rate', 20_000),
                overfit=self.config['data'].get('overfit', False),
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=self.config['num_workers'],
                pin_memory=True,
                shuffle=True,
                collate_fn=self.collate_fn,
                drop_last=True,
            )
        dataset = MatchedSpeechInNoiseDatasetBatched(
            speech_h5_path=self.config['data']['speech_h5_path'],
            noise_h5_path=self.config['data']['noise_h5_path'],
            low_db=self.config['audio_transforms']['low_snr'],
            high_db=self.config['audio_transforms']['high_snr'],
            db_spl=self.config['audio_transforms']['dbspl'],
            batch_size=self.config['hparas']['batch_size'],
            target_keys=self.config['data']['target_keys'],
            blocked_batches=self.config['data'].get('blocked_batches', True),
            signal_augment=self.config['data'].get('signal_augment', False),
            skip_aug_match=self.config['data'].get('skip_aug_match', False),
            clean_percentage=self.config['data'].get('clean_percentage', 0.0),
            overfit=self.config['data'].get('overfit', False),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        if self.dataset_name == "MatchedAudiosetBatched":
            dataset = MatchedSpeechInNoiseDatasetBatched(
                speech_h5_path=self.config['data']['val_speech_h5_path'],
                noise_h5_path=self.config['data']['val_noise_h5_path'],
                low_db=self.config['audio_transforms']['low_snr'],
                high_db=self.config['audio_transforms']['high_snr'],
                db_spl=self.config['audio_transforms']['dbspl'],
                batch_size=self.config['hparas']['batch_size'],
                signal_augment=self.config['data'].get("signal_augment", False),
                skip_aug_match=self.config['data'].get("skip_aug_match", False),
                target_keys=self.config['data'].get("target_keys", None),
                blocked_batches=self.config['data'].get("blocked_batches", True),
                clean_percentage=self.config['data'].get("clean_percentage", 0.0),
                overfit=self.config['data'].get('overfit', False),
            )
        else:
            dataset = MatchedSpeechInNoiseDatasetBatched(
                speech_h5_path=self.config['data']['val_speech_h5_path'],
                noise_h5_path=self.config['data']['val_noise_h5_path'],
                low_db=self.config['audio_transforms']['low_snr'],
                high_db=self.config['audio_transforms']['high_snr'],
                db_spl=self.config['audio_transforms']['dbspl'],
                batch_size=self.config['hparas']['batch_size'],
                target_keys=self.config['data']['target_keys'],
                blocked_batches=self.config['data'].get('blocked_batches', True),
                signal_augment=self.config['data'].get('signal_augment', False),
                skip_aug_match=self.config['data'].get('skip_aug_match', False),
                clean_percentage=self.config['data'].get('clean_percentage', 0.0),
                overfit=self.config['data'].get('overfit', False),
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    # @property
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
    