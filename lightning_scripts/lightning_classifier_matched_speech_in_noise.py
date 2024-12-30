
import torch
from torch import nn
from torchmetrics.classification import Accuracy, BinaryPrecision
import torch.nn.functional as F
import lightning as L

import sys
sys.path.append('./lightning_scripts/')
import robustness.audio_models as architectures
import robustness.audio_functions.audio_transforms as at 
from robustness.audio_functions.jsinV3_loss_functions import jsinV3_multi_task_loss
from robustness.audio_functions.audio_input_representations import AUDIO_INPUT_REPRESENTATIONS

from jsinV3DataLoader_precombined_batched import MatchedSpeechInNoiseDatasetBatched

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
            'final/signal/word_int',
            'final/signal/speaker_int',
            'final/noise/labels_int',
        ]

        self.model = ModelWithFrontEnd(self.audio_rep, self.model)

    
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
        self.log(f"{step_type}_loss", loss.detach(), prog_bar=True)
        
        # calc acc, add acc and task loss to log
        for task, task_loss in task_loss_dict.items():
            task_acc = self.accuracy[step_type][task](logits[task], label_dict[task])
            # format task str for logging: remove 'noise/' or 'signal/' from str
            self.log(f"{step_type}_{task}_loss", task_loss.detach(),
                                            #  on_step=True, on_epoch=False,
                                             prog_bar=True, sync_dist=False if step_type == 'train' else True)
            self.log(f"{step_type}_{task}_acc", task_acc,
                                                        #  on_step=True,
                                                        #   on_epoch=False,
                                                          prog_bar=False, sync_dist=False if step_type == 'train' else True)
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
        opt = getattr(torch.optim, self.config['hparas']['optimizer'])
        self.optimizer = opt(self.model.parameters(),  **self.config['hparas']['optimizer_kwargs'])     
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['hparas']['step_lr']) 
              
        return [self.optimizer],   {
                    'scheduler': self.schedule,  # The LR scheduler instance (required)
                    'interval': 'epoch',  # The unit of the scheduler's step size
                }

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
            labels[label_key] = torch.concat([label_set[label_key] for label_set in targets])
        return audio, labels

    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        dataset = MatchedSpeechInNoiseDatasetBatched(speech_h5_path=self.config['data']['speech_h5_path'],
                                                     noise_h5_path=self.config['data']['noise_h5_path'],
                                                     low_db=self.config['audio_transforms']['low_snr'],
                                                     high_db=self.config['audio_transforms']['high_snr'],
                                                     db_spl=self.config['audio_transforms']['dbspl'],
                                                     batch_size=self.config['hparas']['batch_size'],
                                                     target_keys=self.config['data']['target_keys'],
                                                     overfit=self.config['data'].get('overfit', False)
                                                     )
        
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            # persistent_workers=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        return train_dataloader
    
    def val_dataloader(self):
        dataset = MatchedSpeechInNoiseDatasetBatched(speech_h5_path=self.config['data']['val_speech_h5_path'],
                                                     noise_h5_path=self.config['data']['val_noise_h5_path'],
                                                     low_db=self.config['audio_transforms']['low_snr'],
                                                     high_db=self.config['audio_transforms']['high_snr'],
                                                     db_spl=self.config['audio_transforms']['dbspl'],
                                                     batch_size=self.config['hparas']['batch_size'],
                                                     target_keys=self.config['data']['target_keys'],

                                                     )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate_fn,

        )
        return dataloader

