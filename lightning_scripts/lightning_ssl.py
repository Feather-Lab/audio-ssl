
import torch
from torch import nn
# import torchvision
import torch.nn.functional as F
import lightning as L

from . import architectures
from .metrics import calculate_accuracy
from .loss_functions import MMCR_Loss as Dual_MMCR_Loss
import audio_ssl.losses as ssl_losses 

from audio_ssl.misc import LARS, CosineWarmupScheduler
from typing import List, Union, Tuple
# from pprint import pprint
# from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler

from .jsinV3DataLoader_precombined_batched import jsinV3_precombined_paired_batched
import robustness.audio_functions.audio_transforms as at 
from robustness.audio_functions.jsinV3DataLoader_precombined import jsinV3_precombined_paired
from robustness.audio_functions.audio_input_representations import AUDIO_INPUT_REPRESENTATIONS

class ModelWithFrontEnd(nn.Module):
    def __init__(self,front_end, model):
        super().__init__()
        self.front_end = front_end
        self.model = model

    def forward(self, x):
        x, _ = self.front_end(x, None)
        feature, out, logits = self.model(x)
        return feature, out, logits    

class LitAudioSSL(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 

        # Init audio transforms 
        self.transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.CombineWithRandomDBSNR(low_snr=config['audio_transforms']['low_snr'],
                                          high_snr=config['audio_transforms']['high_snr']),
                at.DBSPLNormalizeForegroundAndBackground(dbspl=config['audio_transforms']['dbspl']),
                at.UnsqueezeAudio(dim=0) # dim=0 here so batches of audio from dataloader will be (Batch, 1, Time)
            ])

        # Get audio config and init representation 
        self.audio_config = AUDIO_INPUT_REPRESENTATIONS[config['audio_rep']['name']]
        self.audio_rep = at.AudioToAudioRepresentation(**self.audio_config)

        # Get audio model from config kwargs
        self.model = architectures.__dict__[self.config['model']['arch_name']](**self.config['model']['arch_kwargs'])
        
        if config['audio_rep']['on_gpu']:
            # If computing rep on gpu, compose rep and model in same forward pass for convenience
            self.model = ModelWithFrontEnd(self.audio_rep, self.model)
        else:
            # if computing rep on cpu, add rep as last stage of audio transforms
            self.transforms = at.AudioCompose([
                self.transforms,
                self.audio_rep
            ])
        
        # init losses 
        # if torch.distributed.is_initialized():
        distributed = torch.distributed.is_initialized()
        self.ssl_task = self.config['hparas']['ssl_task']
        if self.ssl_task == 'dual':
            self.ssl_loss = Dual_MMCR_Loss(distributed=distributed) # comeback to see if distrubuted needs to be true here 
        else:
            self.ssl_loss = ssl_losses.__dict__[self.config['hparas']['ssl_loss']](**self.config['hparas']['ssl_loss_kwargs'], distributed=distributed)
            # crop batch size for  data loader  by 2 for single-task ssl models 
            self.config['hparas']['batch_size'] =  int(self.config['hparas']['batch_size'] // 2 )
        self.ssl_loss_str = self.config['hparas']['ssl_loss_str'] # str for logs 
        # scaling factor to apply to self-supervised task loss - default is 1.
        self.lambda_ssl = self.config['hparas'].get('lambda_ssl', 1.0)
        self.opt_supervised_task = self.config['model']['arch_kwargs']['supervised']
        if self.opt_supervised_task:
            self.class_loss = nn.CrossEntropyLoss()

    def _step(self, batch, batch_idx, step_type):
        spec_11, spec_12, spec_21, spec_22, labels_1, labels_2 = batch

        # pass pairs through model 
        _, out_11, logits_11 = self.model(spec_11)
        _, out_12, logits_12 = self.model(spec_12)
        _, out_21, logits_21 = self.model(spec_21)
        _, out_22, logits_22 = self.model(spec_22)

        ## concat reps based on task 
        if self.ssl_task == 'dual':
            # concat pairs with same equivariances and get dual mmcr loss
            outs_1 = torch.cat([out_11, out_21], dim=0)
            outs_2 = torch.cat([out_12, out_22], dim=0)
            loss_ssl = self.ssl_loss(outs_1, outs_2)

        else:
            if self.ssl_task == 'word':
                # group word pairs as augmentations
                # stack 11 and 12 as augmentations along dim 1 
                outs_1 =  torch.stack([out_11, out_12], dim=1)
                # stack 21 and 22 as augmentations along dim 1 
                outs_2 =  torch.stack([out_21, out_22], dim=1)
                # concat 1x and 2x along batch dim -> b x n_aug x feats 
                outs = torch.cat([outs_1, outs_2])

            elif self.ssl_task == 'audioset':
                # group audioset pairs as augmentations
                # stack 11 and 21 as augmentations along dim 1 
                outs_1 =  torch.stack([out_11, out_21], dim=1)
                # stack 12 and 22 as augmentations along dim 1 
                outs_2 =  torch.stack([out_12, out_22], dim=1)
                # stack x1 and x2 along batch dimensions 
                outs = torch.cat([outs_1, outs_2])

            loss_ssl, _  = self.ssl_loss(outs) 

        self.log(f"{step_type}_{self.ssl_loss_str}_loss", loss_ssl.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        class_loss = 0.0
        if self.opt_supervised_task:
        # get classification loss
            class_loss_11 = self.class_loss(logits_11, labels_1)
            class_loss_12 = self.class_loss(logits_12, labels_1)
            class_loss_21 = self.class_loss(logits_21, labels_2)
            class_loss_22 = self.class_loss(logits_22, labels_2)
            class_loss = class_loss_11 + class_loss_12 + class_loss_21 + class_loss_22
            class_loss = class_loss / 4.0

            # calc acc 
            acc = 0 
            acc += calculate_accuracy(logits_11, labels_1).item()
            acc += calculate_accuracy(logits_12, labels_1).item()
            acc += calculate_accuracy(logits_21, labels_2).item()
            acc += calculate_accuracy(logits_22, labels_2).item()
            acc /= 4.0  

            self.log(f"{step_type}_class_acc", acc, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_class_loss", class_loss.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        total_loss = self.lambda_ssl * loss_ssl + class_loss
        self.log(f"{step_type}_total_loss", total_loss.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)


        # add acc to log 
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def on_train_epoch_end(self):
        self.train_dataloader.dataset._rotate_splits()
        print(f"Updated rotation: {self.train_dataloader.dataset.rotate_index}")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        # Optimizer
        if self.config['hparas']['optimizer'] == "LARS":
            # lr = self.config['hparas']['lr'] * self.config['hparas']['batch_size']  / 256 
            # if self.ssl_task == 'word' or self.ssl_task == 'audioset':
            #     lr = lr * 2 # batch size is double here 
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
                batch_size=self.config['hparas']['batch_size'], # is scaled to per-device batch size
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
            self.optimizer = opt(self.model.parameters(), lr=self.config['hparas']['lr'])      
        return [self.optimizer]

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
        """
        Logic for generating views of speech and noise egs. 
        """
        batch = batch[0] # unbox wrapper added by dataloader 
        signal_11, signal_12, signal_21, signal_22 = [], [], [], []
        target_1 = batch[-2] # labels already collated 
        target_2 = batch[-1] # labels already collated 
        # convert labels to torch tensors 
        if isinstance(target_1, dict):
            for task_key, task_labels in target_1.items():
                target_1[task_key] = torch.from_numpy(task_labels)
        if isinstance(target_2, dict):
            for task_key, task_labels in target_2.items():
                target_2[task_key] = torch.from_numpy(task_labels)
        else:
            target_1 = torch.from_numpy(target_1) 
            target_2 = torch.from_numpy(target_2) 
        # convert signal and noise into signal
        for (signal_1, signal_2, noise_1, noise_2) in  zip(*batch[:4]):
            sig_11, _ = self.transforms(signal_1, noise_1)
            sig_12, _ = self.transforms(signal_1, noise_2)
            sig_21, _ = self.transforms(signal_2, noise_1)
            sig_22, _ = self.transforms(signal_2, noise_2)
            # dummy handle noise-only signals:
            sig_11 = sig_12 if sig_11 is None else sig_11
            sig_12 = sig_11 if sig_12 is None else sig_12
            sig_21 = sig_22 if sig_21 is None else sig_21
            sig_22 = sig_21 if sig_22 is None else sig_22

            signal_11.append(sig_11)
            signal_12.append(sig_12)
            signal_21.append(sig_21)
            signal_22.append(sig_22)

        signal_11 = torch.cat(signal_11).unsqueeze(1) # add back channel dim
        signal_12 = torch.cat(signal_12).unsqueeze(1) # add back channel dim
        signal_21 = torch.cat(signal_21).unsqueeze(1) # add back channel dim
        signal_22 = torch.cat(signal_22).unsqueeze(1) # add back channel dim

        return signal_11, signal_12, signal_21, signal_22, target_1, target_2
        
    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        dataset = jsinV3_precombined_paired_batched(root=self.config['data']['root'],
                                            train=True,
                                            batch_size=self.config['hparas']['batch_size'],
                                            transform=self.transforms)
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
        dataset = jsinV3_precombined_paired_batched(root=self.config['data']['root'],
                                            train=False,
                                            batch_size=self.config['hparas']['batch_size'],
                                            transform=self.transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return dataloader

    # @property
    def total_training_steps(self) -> int:
        dataset_size = len(self.train_dataloader())
        # pprint(vars(self.trainer))
        # print(self.trainer.num_gpus)

        # num_devices = self.trainer.devices if self.trainer.devices else self.trainer.num_processes
        num_devices = self.config['num_gpus']
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_training_steps