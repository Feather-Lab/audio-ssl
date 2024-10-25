
import torch
from torch import nn
# import torchvision
import torch.nn.functional as F
import lightning as L
import os, sys

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "lightning_scripts"))
import architectures
from metrics import calculate_accuracy
import losses as ssl_losses
# import audio_ssl.losses as ssl_losses 

from audio_ssl.misc import LARS, CosineWarmupScheduler
from typing import List, Union, Tuple
# from pprint import pprint

from jsinV3DataLoader_precombined_batched import jsinV3_precombined_paired_batched
import robustness.audio_functions.audio_transforms as at 
from robustness.audio_functions.audio_input_representations import AUDIO_INPUT_REPRESENTATIONS

class ModelWithFrontEnd(nn.Module):
    def __init__(self,front_end, model):
        super().__init__()
        self.front_end = front_end
        self.model = model

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        x, _ = self.front_end(x, None)
        if with_latent:
            return self.model.f(x,  with_latent=with_latent, fake_relu=fake_relu, no_relu=no_relu)
        else:
            feature, out, logits = self.model(x)
            return feature, out, logits    


class LitAudioSSL(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 

        # Init audio transforms 
        if self.config['audio_transforms'].get('crop', False): # crop will be string name of crop class
            self.transforms = at.AudioCompose([
                    at.AudioToTensor(),
                    at.__dict__[self.config['audio_transforms']['crop']](**self.config['audio_transforms']['crop_kwrgs']),
                    at.CombineWithRandomDBSNR(low_snr=config['audio_transforms']['low_snr'],
                                            high_snr=config['audio_transforms']['high_snr']),
                    at.DBSPLNormalizeForegroundAndBackground(dbspl=config['audio_transforms']['dbspl']),
                    at.UnsqueezeAudio(dim=0) # dim=0 here so batches of audio from dataloader will be (Batch, 1, Time)
                ])
        else:
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
        self.distributed = torch.distributed.is_initialized()
        self.ssl_task = self.config['hparas']['ssl_task']
        self.ssl_loss = self.get_loss()

        # scaling factor to apply to self-supervised task loss - default is 1.
        self.lambda_ssl = self.config['hparas'].get('lambda_ssl', 1.0)
        self.opt_supervised_task = self.config['model']['arch_kwargs']['supervised']
        if self.opt_supervised_task:
            self.class_loss = nn.CrossEntropyLoss()

        # get lower bound for MMCR task 

    def _step(self, batch, batch_idx, step_type):
        spec_11, spec_12, spec_21, spec_22, labels_1, labels_2 = batch

        # pass pairs through model 
        _, out_11, logits_11 = self.model(spec_11)
        _, out_12, logits_12 = self.model(spec_12)
        _, out_21, logits_21 = self.model(spec_21)
        _, out_22, logits_22 = self.model(spec_22)

        ## concat reps based on task 
        if self.ssl_task == 'dual':
            # concat is handled in the paired loss function 
            loss_ssl = self.ssl_loss(out_11, out_12, out_21, out_22)

        else:
            if self.ssl_task == 'word':
                # group word pairs as augmentations
                # cat 11 and 21 as batch view 1 along dim 0 
                z_1 =  torch.cat([out_11, out_21], dim=0)
                # cat 12 and 22 as batch view 2 along dim 0 
                z_2 =  torch.cat([out_12, out_22], dim=0)

            elif self.ssl_task == 'audioset':
                # group audioset pairs as augmentations
                # cat 11 and 21 as batch view 1 along dim 0 
                z_1 =  torch.cat([out_11, out_12], dim=0)
                # cat 21 and 22 as batch view 2 along dim 0 
                z_2 =  torch.cat([out_21, out_22], dim=0)
            # z_1 and z_2 are the different views
            loss_ssl = self.ssl_loss(z_1, z_2) 

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

        if 'mmcr' in self.ssl_loss_str:
            # log pretraining percent error (Eq. 4 in https://arxiv.org/pdf/2406.09366):
            # (lower_bound - nuclear_norm_C) / lower_bound
            # lower bound is sqrt(p * min(d,p)); p=points d=dimension
            # Sum because loss is already negative 
            ppe = (self.mmcr_lower_bound + loss_ssl.detach()) / self.mmcr_lower_bound
            self.log(f"{step_type}_ppe", ppe, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # add acc to log 
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    # don't need this anymore - maintaining temporarily in case it becomes useful 
    # def on_train_epoch_end(self): 
    #     self.train_dataloader.dataset._rotate_splits()
    #     print(f"Updated rotation: {self.train_dataloader.dataset.rotate_index}")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        # Test step only for JSIN eval 
        spec_11, spec_12, spec_21, spec_22, labels_1, labels_2 = batch
                # pass pairs through model 
        _, _, logits_11 = self.model(spec_11)
        _, _, logits_12 = self.model(spec_12)
        _, _, logits_21 = self.model(spec_21)
        _, _, logits_22 = self.model(spec_22)

        class_loss = 0.0
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

        self.log(f"test_class_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"test_class_loss", class_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return class_loss


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
            # if any([sig.sum() == 0 for sig in [signal_1, signal_2, noise_1, noise_2]]):
            #     continue 
            sig_11, _ = self.transforms(signal_1, noise_1)
            sig_12, _ = self.transforms(signal_1, noise_2)
            sig_21, _ = self.transforms(signal_2, noise_1)
            sig_22, _ = self.transforms(signal_2, noise_2)
            # # dummy handle noise-only signals:
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
        num_devices = self.config['num_gpus']
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return int(self.trainer.max_steps)
        return int(max_estimated_steps)

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_training_steps
    
    @property
    def mmcr_lower_bound(self) -> int:
        # precompute mmcr lower bound as prop 3.3 from https://arxiv.org/pdf/2406.09366
        p = torch.tensor(self.config['hparas']['global_batch_size'])
        d = torch.tensor(self.config['model']['arch_kwargs']['projector_dims'][-1])
        return torch.sqrt(p * torch.min(p, d))

    def get_loss_fn(self, loss_fn_name, loss_kwargs):
        loss_fn = ssl_losses.__dict__[loss_fn_name]
        return loss_fn(**loss_kwargs, distributed=self.distributed) if loss_kwargs else loss_fn(distributed=self.distributed)

    def get_loss(self):
        self.ssl_loss_str = self.config['hparas']['ssl_loss_str'] # str for logs 
        loss_kwargs = self.config['hparas'].get('ssl_loss_kwargs', None) 
        if 'paired' in  self.ssl_loss_str:
            loss_fn_inv_kwargs = loss_kwargs.get('loss_fn_inv_kwargs', None) 
            loss_fn_eq_kwargs = loss_kwargs.get('loss_fn_eq_kwargs', None) 
            loss_fn_inv = self.get_loss_fn(loss_kwargs['loss_fn_inv'], loss_fn_inv_kwargs)
            loss_fn_eq = self.get_loss_fn(loss_kwargs['loss_fn_eq'], loss_fn_eq_kwargs)
            paired_loss =  ssl_losses.__dict__[self.config['hparas']['ssl_loss']]
            ssl_loss = paired_loss(loss_fn_inv=loss_fn_inv, loss_fn_eq=loss_fn_eq, lmda=loss_kwargs['lmda'])
        else:
            ssl_loss = self.get_loss_fn(self.config['hparas']['ssl_loss'], loss_kwargs)
        return ssl_loss