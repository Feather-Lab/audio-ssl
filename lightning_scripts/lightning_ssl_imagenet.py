
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
import imagenet_dataset as img_ds 


class LitImageSSL(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 

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

    def _step(self, batch, batch_idx, step_type):
        [img_1, img_2], labels  = batch
        # pass pairs through model 
        _, out_1, logits_1 = self.model(img_1)
        _, out_2, logits_2 = self.model(img_2)

        loss_ssl = self.ssl_loss(out_1, out_2) 

        self.log(f"{step_type}_{self.ssl_loss_str}_loss", loss_ssl.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        class_loss = 0.0
        if self.opt_supervised_task:
        # get classification loss
            class_loss_1 = self.class_loss(logits_1, labels)
            class_loss_2 = self.class_loss(logits_2, labels)
            class_loss = (class_loss_1 + class_loss_2 ) / 2.0

            # calc acc 
            acc = 0 
            acc += calculate_accuracy(logits_1, labels).item()
            acc += calculate_accuracy(logits_2, labels).item()
            acc /= 2.0

            self.log(f"{step_type}_class_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{step_type}_class_loss", class_loss.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        total_loss = self.lambda_ssl * loss_ssl + class_loss

        self.log(f"{step_type}_total_loss", total_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if 'mmcr' in self.ssl_loss_str:
            ppe = (self.mmcr_lower_bound + loss_ssl.detach()) / self.mmcr_lower_bound
            self.log(f"{step_type}_ppe", ppe, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # add acc to log 
        return total_loss

    def _eval_step(self, batch, batch_idx, step_type):
        # Test step only for JSIN eval 
        img, labels  = batch
        # pass pairs through model 
        _, out_1, logits_1 = self.model(img)

        class_loss = self.class_loss(logits_1, labels)
        acc = calculate_accuracy(logits_1, labels).item()

        self.log(f"{step_type}_class_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{step_type}_class_loss", class_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return class_loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, "test")

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
            lr = self.config['hparas']['lr'] * self.config['hparas']['global_batch_size'] / 256 
            opt = getattr(torch.optim, self.config['hparas']['optimizer'])
            self.optimizer = opt(self.model.parameters(), lr=lr)      
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
        img1, img2, labels = [], [], []
        for img, label in batch:
            if len(img) == 2:
                img1.append(img_ds.imagenet_denormalizer(img[0]))
                img2.append(img_ds.imagenet_denormalizer(img[1]))
            else:
                img1.append(img_ds.imagenet_denormalizer(img))
            labels.append(label)
        labels = torch.tensor(labels)
        img_1 = torch.stack(img1, dim=0)
        if len(img2) > 0:
            img_2 = torch.stack(img2, dim=0)
            return [img_1, img_2], labels
        else:
            return img_1, labels

    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        dataset = img_ds.Zip_ImageFolder(
                    zip_path=img_ds.imagenet_100_path + "train.zip",
                    root=img_ds.imagenet_100_path + "train/",
                    transform=img_ds.Barlow_Transform(),
                )
        
        dataloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.config['hparas']['batch_size'],
                            num_workers=self.config['num_workers'], 
                            pin_memory=True,
                            # persistent_workers=True,
                            shuffle=True,
                            collate_fn=self.collate_fn
                        )
        return dataloader

    def val_dataloader(self):
        dataset = img_ds.Zip_ImageFolder(
                    zip_path=img_ds.imagenet_100_path + "val.zip",
                    root=img_ds.imagenet_100_path + "val/",
                    transform=img_ds.ImageNetValTransform(),
                )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['hparas']['batch_size'],
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

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps and self.trainer.max_steps != -1:
            return int(self.trainer.max_steps)
        return int(max_estimated_steps)

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_warmup_steps
    
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
            ssl_loss = paired_loss(loss_fn_inv=loss_fn_inv, loss_fn_eq=loss_fn_eq, lmda=loss_kwargs['lmda'])
        else:
            ssl_loss = self.get_loss_fn(self.config['hparas']['ssl_loss'], loss_kwargs)
        return ssl_loss