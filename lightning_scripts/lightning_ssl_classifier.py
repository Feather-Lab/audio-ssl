import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from default_paths import JSIN_PATH, require_path
import torch 
import torch.nn as nn 
import lightning as L
import re
from typing import Union
from lightning_ssl_matched_speech_in_noise import LitAudioSSL
from whisper_encoder_arch import get_whisper_encoder_layer_sizes
from lightning_classifier_matched_speech_in_noise import LitWordAudioSetModel as LitAudioSupervised
from optimizers import LARS, CosineWarmupScheduler
from jsinV3DataLoader_precombined_batched import jsinV3_precombined_all_signals
from torchmetrics.classification import Accuracy, BinaryPrecision
from robustness.audio_functions.jsinV3_loss_functions import jsinV3_multi_task_loss
import robustness.audio_functions.audio_transforms as at

class SSLClassifier(L.LightningModule):
    def __init__(self, config, ckpt_path, layer_out, supervised_backbone=False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.layer_out = layer_out
        self.byola_arch = False
        self.act_hook_dict = {}

        # init the pretrained LightningModule
        # Set strict to false to ignore loading in pre-trained classifier 
        if supervised_backbone:
            self.feature_extractor = LitAudioSupervised.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, strict=False).eval()
            self.config['model']['arch_kwargs']['backbone'] = config['model']['arch_name']

        else:
            self.feature_extractor = LitAudioSSL.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, strict=False).eval()
        self.feature_extractor = torch.compile(self.feature_extractor)
        self.feature_extractor.freeze()
        # softcode size dict at some point 
        self.time_avg_rep = config['model']['arch_kwargs'].get('time_average', True)
        self.is_whisper = config['model']['arch_kwargs'].get('backbone') == 'whisper' or 'whisper' in config['model'].get('arch_name', '')
        self.crop_audio = config.get('crop_audio', False)
        self.no_avgpool = config['model']['arch_kwargs'].get('no_avgpool', False)
        if self.crop_audio:
            self.audio_crop = at.CenterCropForegroundBackground(signal_size=40_000, crop_length=20_000) # random crop to 1 second, centered on word

        self.transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.CombineWithRandomDBSNR(low_snr=-10,
                                        high_snr=10),
                at.DBSPLNormalizeForegroundAndBackground(dbspl=60),
                at.UnsqueezeAudio(dim=0) # dim=0 here so batches of audio from dataloader will be (Batch, 1, Time)
            ])

        self.test_transforms = at.AudioCompose([
                at.AudioToTensor(),
                at.CenterCropForegroundBackground(signal_size=40_000, crop_length=40_000),
                at.CombineWithRandomDBSNR(low_snr=-10,
                                        high_snr=10),
                at.DBSPLNormalizeForegroundAndBackground(dbspl=60),
                at.UnsqueezeAudio(dim=0) # dim=0 here so batches of audio from dataloader will be (Batch, 1, Time)
            ])
        
        if self.is_whisper:
            encoder_kwargs = config['model']['arch_kwargs'].get('encoder_kwargs', {})
            layer_size_dict = get_whisper_encoder_layer_sizes(encoder_kwargs, time_average=self.time_avg_rep)
        elif config['model']['arch_kwargs']['backbone'] == 'kell2018' or 'kell2018' in config['model']['arch_name']:
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
            
        elif config['model']['arch_kwargs']['backbone'] in ['AudioNTT2020']:
            layer_size_dict = {'input_after_preproc': 211,
                                "relu0": 5256960,
                                "relu1": 1310400,
                                "relu2": 323584,
                                "relu3": 98304,
                                "relu4": 98304,
                                'final': 2048}
            self.byola_arch = True          
            self.register_relu_hook_by_index()

        elif config['model']['arch_kwargs']['backbone'] in ['resnet18', 'resnet_multi_task18']:
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
            if self.no_avgpool:
                layer_size_dict['avgpool'] = layer_size_dict['layer4']
        else:
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
                                    'avgpool': 2048,
                                    'final': 2048}
            else:
                layer_size_dict = {'input_after_preproc': 82290,
                                    'conv1': 1322880,
                                    'bn1': 1322880,
                                    'conv1_relu1': 1322880,
                                    'maxpool1': 332416,
                                    'layer1': 332416,
                                    'layer2': 169344,
                                    'layer3': 89600,
                                    'layer4': 186368,
                                    'avgpool': 2048,
                                    'final': 2048}
                if self.no_avgpool:
                    layer_size_dict['avgpool'] = layer_size_dict['layer4']
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
        if config['model'].get('with_dropout', False):
            self.dropout = nn.Dropout(p=0.5) # TODO: softcode p
        else:
            self.dropout = False 

        self.multi_task_loss = jsinV3_multi_task_loss(task_loss_params=config['hparas']['task_loss_params'],
                                                      batch_size=None,
                                                    #   reduction='none'
                                                    )

        # Initialize accuracy metrics - handle both single-task (int) and multi-task (dict) cases
        if isinstance(num_classes, dict):
            self.accuracy = torch.nn.ModuleDict({task_key: BinaryPrecision() if 'noise' in task_key else Accuracy(task="multiclass", num_classes=num_classes) 
                            for task_key,num_classes in self.config['model']['arch_kwargs']['num_classes'].items()})
        else:
            # Single-task case (e.g., NSynth) - create accuracy metric with correct number of classes
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes) 
    
    # setup activation hook for byol-a architecture

    def get_activation_hook(self, name):
        def hook(model, input, output):
            self.act_hook_dict[name] = output.detach()
        return hook
    
    def register_relu_hook_by_index(self):
        self.act_hook_dict.clear()
        relu_index = int(re.search(r"\d+", self.layer_out).group())
        relus = [m for m in self.feature_extractor.modules() if isinstance(m, nn.ReLU)]
        if relu_index >= len(relus):
            raise IndexError(f"Model only has {len(relus)} ReLU layers. Got index {relu_index}.")
        relu_layer = relus[relu_index]
        relu_layer.register_forward_hook(self.get_activation_hook(f"relu{relu_index}"))

    def forward(self, x):
        with torch.no_grad():
            if self.byola_arch:
                _, _, _ = self.feature_extractor.model(x,  with_latent=False, fake_relu=False)
                activations = self.act_hook_dict[self.layer_out]
            else:
                predictions, rep, all_outputs = self.feature_extractor.model(x,  with_latent=True, fake_relu=False)
                activations = all_outputs[self.layer_out]
            if self.time_avg_rep:
                activations = activations.mean(dim=-1).reshape(activations.shape[0], -1)
            else:
                activations = activations.reshape(activations.shape[0], -1)
                # time average then flatten
            activations = activations.detach()
        if self.dropout:
            activations = self.dropout(activations)
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
            else:
                noise = None 
            if self.crop_audio:
                signal, noise = self.audio_crop(signal, noise)
                ## Pad back to full length 
                signal = at.pad_or_trim_to_len(signal, 40000, mode="both")
                if noise is not None:
                    noise = at.pad_or_trim_to_len(noise, 40000, mode="both")
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
            signal, _ = self.test_transforms(signal, None)
            if signal is None:
                # Signal was none & has null label class 
                signal = torch.zeros(1,40000)
            signals.append(signal)
        signals = torch.cat(signals).unsqueeze(1) # add back channel dim
        return signals, labels  

    
    def train_dataloader(self):
        # set train dataloader as attr so we can rotate examples every epoch 
        dataset = jsinV3_precombined_all_signals(root=str(require_path(JSIN_PATH, "COCHDNN_JSIN_DIR", "JSIN/WSN dataset")),
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
        dataset = jsinV3_precombined_all_signals(root=str(require_path(JSIN_PATH, "COCHDNN_JSIN_DIR", "JSIN/WSN dataset")),
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
    