
import torch
from torch import nn
from torchmetrics.classification import Accuracy, BinaryPrecision
import torch.nn.functional as F
import lightning as L

import sys
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
            'final/noise/labels_binary_via_int',
        ]

        self.model = ModelWithFrontEnd(self.audio_rep, self.model)

    
        self.multi_task_loss = jsinV3_multi_task_loss(task_loss_params=config['hparas']['task_loss_params'],
                                                      batch_size=config['hparas']['batch_size']//4)
        # get accuracy metrics per task - requires module dict for torchmetrics 
        self.train_accuracy = torch.nn.ModuleDict({task_key: BinaryPrecision() if 'binary' in task_key else Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_params']['num_classes'].items()}) 
        
        self.val_accuracy = torch.nn.ModuleDict({task_key: BinaryPrecision() if 'binary' in task_key else Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_params']['num_classes'].items()}) 
        
        self.accuracy = {'train': self.train_accuracy, 'val': self.val_accuracy}

    def _step(self, batch, batch_idx, step_type):
        spec_11, spec_12, spec_21, spec_22, target_11, target_12, target_21 , target_22 = batch 
        ## Permute dims (1, batch, time) -> (batch, 1, time)
        spec_11 = spec_11.permute(1,0,2)
        spec_12 = spec_12.permute(1,0,2)
        spec_21 = spec_21.permute(1,0,2)
        spec_22 = spec_22.permute(1,0,2)

        # logits will be dict - keys for each task
        logits_11 = self.model(spec_11)
        logits_12 = self.model(spec_12)
        logits_21 = self.model(spec_21)
        logits_22 = self.model(spec_22)
        
        # get classification loss for each set
        total_loss = 0. 
        total_task_loss_dict = {task_key:0 for task_key in logits_11.keys()}
        total_task_acc_dict = {task_key:0 for task_key in logits_11.keys()}
        
        for logits, label_dict in zip([logits_11, logits_12, logits_21, logits_22], [target_11, target_12, target_21 , target_22]):
            for task, labels in label_dict.items():
                # remove extra dims from labels
                label_dict[task] = labels.squeeze()
            loss, task_loss_dict = self.multi_task_loss(logits, label_dict, return_indiv_loss=True)
            total_loss += loss 

            # update task loss and task accuracy per combination 
            for task in task_loss_dict.keys():
                total_task_loss_dict[task] += task_loss_dict[task]
                total_task_acc_dict[task] += self.accuracy[step_type][task](logits[task], label_dict[task])

        # add task loss to log
        for task, total_task_loss in total_task_loss_dict.items():
            total_task_loss = total_task_loss.detach() / 4.  # account for each batch 
            self.log(f"{step_type}_{task}_loss", total_task_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            task_acc = total_task_acc_dict[task] / 4
            self.log(f"{step_type}_{task}_acc", task_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        total_loss = total_loss / 4. 
        self.log(f"{step_type}_loss", total_loss.detach(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
  

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        # Optimizer
        opt = getattr(torch.optim, self.config['hparas']['optimizer'])
        self.optimizer = opt(self.model.parameters(), lr=self.config['hparas']['lr'])     
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['hparas']['step_lr']) 
        return [self.optimizer], [self.schedule]

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
        batch = batch[0] # unbox wrapper added by dataloader 
        signals = []
        labels = batch[-1] # labels already collated 
        # convert labels to torch tensors 
        if isinstance(labels, dict):
            for task_key, task_labels in labels.items():
                labels[task_key] = torch.from_numpy(task_labels)
        else:
            labels = torch.from_numpy(labels) 
        # convert signal and noise into signal
        for (signal, noise) in  zip(*batch[:2]):
            signal, _ = self.transforms(signal, noise)
            signals.append(signal)
        signals = torch.cat(signals).unsqueeze(1) # add back channel dim
        return signals, labels 

    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        dataset = MatchedSpeechInNoiseDatasetBatched(speech_h5_path=self.config['data']['speech_h5_path'],
                                                     noise_h5_path=self.config['data']['noise_h5_path'],
                                                     low_db=self.config['audio_transforms']['low_snr'],
                                                     high_db=self.config['audio_transforms']['high_snr'],
                                                     db_spl=self.config['audio_transforms']['dbspl'],
                                                     batch_size=self.config['hparas']['batch_size'],
                                                     target_keys=self.config['data']['target_keys'],
                                                     )
        
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.config['num_workers'], 
            pin_memory=True,
            # persistent_workers=True,
            shuffle=False,
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
        )
        return dataloader

