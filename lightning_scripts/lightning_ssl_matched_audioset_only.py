
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import os, sys
import logging

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "lightning_scripts"))
import architectures
from torchmetrics.classification import Accuracy, BinaryPrecision
import losses as ssl_losses

from audio_ssl.misc import LARS, CosineWarmupScheduler
from typing import List, Union, Tuple

from jsinV3DataLoader_precombined_batched import MatchedAudiosetBatched, MatchedSpeechInNoiseDatasetBatched
import robustness.audio_functions.audio_transforms as at
from robustness.audio_functions.jsinV3_loss_functions import jsinV3_multi_task_loss
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

        # Build the waveform-to-representation frontend used by the encoder.
        self.audio_config = AUDIO_INPUT_REPRESENTATIONS[config['audio_rep']['name']]
        self.audio_rep = at.AudioToAudioRepresentation(**self.audio_config)

        # Get audio model from config kwargs
        self.model = architectures.__dict__[self.config['model']['arch_name']](**self.config['model']['arch_kwargs'])

        if 'resnet' in self.config['model']['arch_kwargs']['backbone']:
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
        elif 'kell2018' in self.config['model']['arch_kwargs']['backbone']:
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

        # If computing rep on gpu, compose rep and model in same forward pass for convenience
        self.model = ModelWithFrontEnd(self.audio_rep, self.model)

        # init losses 
        # if torch.distributed.is_initialized():
        self.distributed = torch.distributed.is_initialized()
        self.ssl_task = self.config['hparas']['ssl_task']
        self.ssl_loss = self.get_loss()

        # scaling factor to apply to self-supervised task loss - default is 1.
        self.lambda_ssl = self.config['hparas'].get('lambda_ssl', 1.0)
        self.skip_pairing = self.config['hparas'].get('skip_pairing', False)
        self.opt_supervised_task = self.config['model']['arch_kwargs']['supervised']
        if self.opt_supervised_task:
            self.multi_task_loss = jsinV3_multi_task_loss(task_loss_params=config['hparas']['task_loss_params'],
                                                      batch_size=None,
                                                    #   reduction='none'
                                                    )
            self.metrics = torch.nn.ModuleDict({task_key: BinaryPrecision() if 'noise' in task_key else Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_kwargs']['num_classes'].items()}) 
        
        self.init_train_log = True 

    def _step(self, batch, batch_idx, step_type):
        if batch_idx == 0:
            logging.getLogger('sox').setLevel(logging.ERROR)
        if self.opt_supervised_task: 
            [spec_11, spec_12, spec_21, spec_22], [labels_11, labels_12, labels_21, labels_22] = batch
        else:
            spec_11, spec_12, spec_21, spec_22  = batch
        ## Permute dims (1, batch, time) -> (batch, 1, time)
        # pass pairs through model 
        _, out_11, logits_11 = self.model(spec_11)
        _, out_12, logits_12 = self.model(spec_12)
        if not self.skip_pairing:
            _, out_21, logits_21 = self.model(spec_21)
            _, out_22, logits_22 = self.model(spec_22)
        ## concat reps based on task 
        if self.ssl_task == 'dual':
            # concat is handled in the paired loss function 
            loss_ssl, inv_loss, eq_loss = self.ssl_loss(out_11, out_12, out_21, out_22)
            if self.loss_kwargs.get('return_both_losses', False):
                # unpack invariant losses for foregrounds and backgrounds 
                inv_loss, inv_fg_loss, inv_bg_loss = inv_loss # is tuple with total, fg, bg 
                self.log(f"{step_type}_inv_fg_loss", inv_fg_loss.detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
                self.log(f"{step_type}_inv_bg_loss", inv_bg_loss.detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
                # unpack equivariant losses for foregrounds and backgrounds 
                eq_loss, eq_fg_loss, eq_bg_loss = eq_loss # is tuple with total, fg, bg 
                self.log(f"{step_type}_eq_fg_loss", eq_fg_loss.detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
                self.log(f"{step_type}_eq_bg_loss", eq_bg_loss.detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

            self.log(f"{step_type}_inv_loss", inv_loss.detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log(f"{step_type}_eq_loss", eq_loss.detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        else:
            if self.ssl_task == 'word':
                # group word pairs as augmentations
                # cat 11 and 21 as batch view 1 along dim 0 
                if self.skip_pairing:
                    z_1 = out_11
                    z_2 = out_12
                else:
                    z_1 =  torch.cat([out_11, out_21], dim=0)
                    # cat 12 and 22 as batch view 2 along dim 0 
                    z_2 =  torch.cat([out_12, out_22], dim=0)

            elif self.ssl_task == 'audioset':
                # group audioset pairs as augmentations
                if self.skip_pairing:
                    z_1 = out_11
                    z_2 = out_21
                else:
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
            class_loss_11, task_loss_11 = self.multi_task_loss(logits_11, labels_11, return_indiv_loss=True)
            class_loss_12, task_loss_12 = self.multi_task_loss(logits_12, labels_12, return_indiv_loss=True)
            if not self.skip_pairing:
                class_loss_21, task_loss_21  = self.multi_task_loss(logits_21, labels_21, return_indiv_loss=True)
                class_loss_22, task_loss_22 = self.multi_task_loss(logits_22, labels_22, return_indiv_loss=True)
                total_class_loss = (class_loss_11 + class_loss_12 + class_loss_21 + class_loss_22) / 4.0 
            else: 
                total_class_loss = (class_loss_11 + class_loss_12) / 2.0 

            self.log(f"{step_type}_total_class_loss", total_class_loss.detach(), prog_bar=True, sync_dist=True)

            for task, metric in self.metrics.items():

                task_loss = task_loss_11[task] + task_loss_12[task]
                if not self.skip_pairing:
                    # Add acc per task 
                    task_loss += task_loss_21[task] + task_loss_22[task]
                    task_loss = task_loss / 4.0

                else:
                    task_loss = task_loss / 2.0
                self.log(f"{step_type}_{task}_loss", task_loss.detach(), prog_bar=True, sync_dist=True)

                acc = 0
                acc += metric(logits_11[task], labels_11[task]).item()
                acc += metric(logits_12[task], labels_12[task]).item()
                if not self.skip_pairing:
                    acc += metric(logits_21[task], labels_21[task]).item()
                    acc += metric(logits_22[task], labels_22[task]).item()
                    acc /= 4.0  
                else:
                    acc /= 2.0

                if 'signal' in task:
                    self.log(f"{step_type}_{task}_acc", acc,  prog_bar=False, sync_dist=True)
                else:
                    self.log(f"{step_type}_{task}_prec", acc, prog_bar=False, sync_dist=True)



        total_loss = self.lambda_ssl * loss_ssl + total_class_loss
        self.log(f"{step_type}_total_loss", total_loss.detach(), prog_bar=True, sync_dist=True)

        if 'mmcr' in self.ssl_loss_str:
            # log pretraining percent error (Eq. 4 in https://arxiv.org/pdf/2406.09366):
            # (lower_bound - nuclear_norm_C) / lower_bound
            # lower bound is sqrt(p * min(d,p)); p=points d=dimension
            # Sum because loss is already negative 
            ppe = (self.mmcr_lower_bound + loss_ssl.detach()) / self.mmcr_lower_bound
            self.log(f"{step_type}_ppe", ppe, prog_bar=True, sync_dist=True)

        if 'paired' in self.ssl_loss_str and self.inv_loss_type == "MMCR_Loss" and self.eq_loss_type == "MMCR_Loss":
            inv_ppe = (self.mmcr_lower_bound + inv_loss.detach()) / self.mmcr_lower_bound
            self.log(f"{step_type}_inv_ppe", inv_ppe, prog_bar=True, sync_dist=True)
            eq_ppe = (self.mmcr_lower_bound + eq_loss.detach()) / self.mmcr_lower_bound
            self.log(f"{step_type}_eq_ppe", eq_ppe, prog_bar=True, sync_dist=True)

        # add acc to log 
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        # Optimizer
        if self.config['hparas']['optimizer'] == "LARS":
            if self.config['hparas'].get("num_warmup_steps_or_ratio", False):
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
                lr = self.config['hparas']['lr'] * self.config['hparas']['global_batch_size'] / 256 
                self.optimizer = LARS(
                                self.model.parameters(),
                                lr=lr,
                                weight_decay=1e-6,
                                momentum=0.9,
                                weight_decay_filter=True,
                                lars_adaptation_filter=True,
                            )                        
        else:
            lr = self.config['hparas']['lr'] * self.config['hparas']['global_batch_size'] / 256 
            opt = getattr(torch.optim, self.config['hparas']['optimizer'])
            self.optimizer = opt(self.model.parameters(), lr=lr)      
        return [self.optimizer]

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

    def  on_validation_epoch_start(self):
        if  self.init_train_log:
            ### Hack to log train_total_loss when resuming from checkpoint 
            self.log(f"train_total_loss", 100.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.init_train_log = False 


    def forward(self, x, with_latent=False):
        """
        PL required forward wrapper. Enables calling model in two ways:
        1) standard call in .py scripts
            model = LitAudioSSL(args)
            outs = model(inputs)
        2) inside this lightning module's methods as self (eg in _step)
            outs = self(inputs) # self is self.forward, and is same as self.model.forward 
        """
        return self.model(x, with_latent=with_latent)

    def collate_fn(self, batch):
        batch = batch[0]
        if len(batch) == 2:
            all_audio = []
            for audio in batch[0]:
                all_audio.append(audio.unsqueeze(1))
            labels = []
            for label_set in batch[1]:
                view_labels = {}
                if isinstance(label_set, dict):
                    for key, l in label_set.items():
                        ## TODO: Sanity check this is right
                        l = l.squeeze()
                        l = F.pad(l,
                                                    (0, 527 - l.shape[-1]),
                                                    mode='constant',
                                                    value=0)
                        view_labels[key] = l.squeeze()
                    labels.append(view_labels)
                else:
                    labels.append(label_set.squeeze())
            return all_audio, labels 
        elif len(batch) == 4: # no labels
            return [audio.unsqueeze(1) for audio in batch]
  
    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
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
                                        )
        
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            # persistent_workers=True,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True
        )
        print(f"Rank {self.trainer.local_rank} N training batches {len(train_dataloader)}")

        return train_dataloader
    
    def val_dataloader(self):
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
                                                     )
        print(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate_fn,
            # drop_last=True

        )
        print(f"Rank {self.trainer.local_rank} N validation batches {len(dataloader)}")
        return dataloader

    # @property
    def total_training_steps(self) -> int:
        dataset_size = len(self.train_dataloader())
        if self.skip_pairing:
            dataset_size // 2 # not including pairing 
        num_devices = self.config['num_gpus']
        effective_batch_size = self.trainer.accumulate_grad_batches * max(num_devices, 1)
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps and self.trainer.max_steps != -1:
            return int(self.trainer.max_steps)
        return int(max_estimated_steps)

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_warmup_steps
    
    @property
    def mmcr_lower_bound(self) -> int:
        # precompute mmcr lower bound as prop 3.3 from https://arxiv.org/pdf/2406.09366
        p = torch.tensor(self.config['hparas']['global_batch_size'])
        if (self.ssl_task == 'word' or self.ssl_task == 'audioset') and not self.skip_pairing:
            p *= 2 # account for stacking "paired" examples along batch dimension
        d = torch.tensor(self.config['model']['arch_kwargs']['projector_dims'][-1])
        return torch.sqrt(p * torch.min(p, d))

    def get_loss_fn(self, loss_fn_name, loss_kwargs):
        loss_fn = ssl_losses.__dict__[loss_fn_name]
        return loss_fn(**loss_kwargs, distributed=self.distributed) if loss_kwargs else loss_fn(distributed=self.distributed)

    def get_loss(self):
        self.ssl_loss_str = self.config['hparas']['ssl_loss_str'] # str for logs 
        loss_kwargs = self.config['hparas'].get('ssl_loss_kwargs', None) 
        self.loss_kwargs = loss_kwargs
        self.inv_loss = None
        self.eq_loss = None
        if 'paired' in  self.ssl_loss_str:
            loss_fn_inv_kwargs = loss_kwargs.get('loss_fn_inv_kwargs', None) 
            loss_fn_eq_kwargs = loss_kwargs.get('loss_fn_eq_kwargs', None) 
            loss_fn_inv = self.get_loss_fn(loss_kwargs['loss_fn_inv'], loss_fn_inv_kwargs)
            self.inv_loss_type = loss_kwargs['loss_fn_inv']
            loss_fn_eq = self.get_loss_fn(loss_kwargs['loss_fn_eq'], loss_fn_eq_kwargs)
            self.eq_loss_type = loss_kwargs['loss_fn_eq']
            paired_loss =  ssl_losses.__dict__[self.config['hparas']['ssl_loss']]
            if paired_loss ==  ssl_losses.__dict__['Dual_Paired_Loss']:
                ssl_loss = paired_loss(loss_fn_inv=loss_fn_inv, loss_fn_eq=loss_fn_eq, lmda=loss_kwargs['lmda'], return_both_losses=loss_kwargs.get('return_both_losses', False) )
            else:
                ssl_loss = paired_loss(loss_fn_inv=loss_fn_inv, loss_fn_eq=loss_fn_eq, lmda=loss_kwargs['lmda'])
        else:
            ssl_loss = self.get_loss_fn(self.config['hparas']['ssl_loss'], loss_kwargs)
        return ssl_loss