import torch 
import torch.nn as nn 
import lightning as L
import re
from typing import Union
import whisper 
import numpy as np
import torchaudio
from jsinV3DataLoader_precombined_batched import jsinV3_precombined_all_signals
from torchmetrics.classification import Accuracy, BinaryPrecision
from robustness.audio_functions.jsinV3_loss_functions import jsinV3_multi_task_loss
import robustness.audio_functions.audio_transforms as at
from audio_ssl.misc import LARS, CosineWarmupScheduler

def get_whisper_layer_name_map(whisper_model_name: str) -> tuple[dict[str, int], int]:
    """Return a layer-name map and layer count for a Whisper encoder."""
    whisper_model = whisper.load_model(whisper_model_name)
    num_layers = len(whisper_model.encoder.blocks)
    layer_map = {f"encoder_{idx}": idx for idx in range(num_layers)}
    layer_map.update({f"encoder_block_{idx}": idx for idx in range(num_layers)})
    layer_map.update({f"block_{idx}": idx for idx in range(num_layers)})
    return layer_map, num_layers

class WhisperTransferModule(L.LightningModule):
    def __init__(self, config, ckpt_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 
        
        # Load Whisper model to get dimensions, then extract only encoder
        whisper_model_name = config['model'].get('whisper_model', 'large-v3-turbo')
        whisper_model = whisper.load_model(whisper_model_name)
        
        # Get mel spectrogram parameters from the model
        self.n_mels = whisper_model.dims.n_mels
        
        # Extract only the encoder (decoder not needed for feature extraction)
        self.whisper_encoder = whisper_model.encoder
        
        # Get encoder layer to extract features from
        self.encoder_layer_idx = config['model']['arch_kwargs'].get('encoder_layer', 31)
        
        # Freeze the encoder
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        self.whisper_encoder.eval()
        
        # Set up hook to extract encoder layer features
        self.encoder_activations = {}
        self._register_encoder_hook()
        
        # Calculate classifier input dimension directly from model dimensions
        # Whisper encoder output: (batch, n_audio_ctx, n_audio_state)
        # We only use first 2 seconds = 100 tokens (50 Hz sampling rate)
        # We flatten to: (batch, 100 * n_audio_state)
        n_audio_state = whisper_model.dims.n_audio_state  # feature dimension
        self.n_time_tokens = 100  # 2 seconds * 50 Hz
        self.classifier_input_dim = self.n_time_tokens * n_audio_state
        
        self.crop_audio = config.get('crop_audio', False)
        if self.crop_audio:
            self.audio_crop = at.CenterCropForegroundBackground(signal_size=40_000, crop_length=20_000)
        
        # Whisper expects 16kHz audio, but dataset is typically at 20kHz
        # Resample audio to 16kHz for Whisper
        self.resample_audio = lambda x: torchaudio.functional.resample(x, orig_freq=20_000, new_freq=16_000)

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
        
        # init trainable word classifier  
        num_classes = config['model']['arch_kwargs']['num_classes']
        proj_out_dim = self.classifier_input_dim

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

        self.accuracy = torch.nn.ModuleDict({task_key: BinaryPrecision() if 'noise' in task_key else Accuracy(task="multiclass", num_classes=num_classes) 
                        for task_key,num_classes in self.config['model']['arch_kwargs']['num_classes'].items()}) 
    
    def _register_encoder_hook(self):
        """Register a forward hook on the specified encoder layer."""
        def hook_fn(module, input, output):
            # Store the output of the encoder layer
            # output is a tuple, we want the hidden states
            if isinstance(output, tuple):
                # For transformer blocks, output is typically (hidden_states, ...)
                self.encoder_activations['layer_output'] = output[0].detach()
            else:
                self.encoder_activations['layer_output'] = output.detach()
        
        # Get the encoder layers
        encoder_layers = self.whisper_encoder.blocks
        num_layers = len(encoder_layers)
        
        if self.encoder_layer_idx < 0 or self.encoder_layer_idx >= num_layers:
            raise ValueError(f"Encoder layer index {self.encoder_layer_idx} out of range. "
                           f"Model has {num_layers} encoder layers (0-{num_layers-1})")
        
        # Register hook on the specified layer
        encoder_layers[self.encoder_layer_idx].register_forward_hook(hook_fn)

    def forward(self, mel):
        """
        Forward pass through Whisper encoder and classifier.
        
        Args:
            mel: Mel spectrogram tensor of shape (batch, n_mels, n_frames)
                 Already preprocessed (pad_or_trim and log_mel_spectrogram done in collate_fn)
        """
        with torch.no_grad():
            batch_size = mel.shape[0]
            
            # Forward through encoder (this will trigger the hook)
            encoder_output = self.whisper_encoder(mel)
            
            # Get activations from the hook
            if 'layer_output' in self.encoder_activations:
                layer_activations = self.encoder_activations['layer_output']
            else:
                # Fallback: use full encoder output
                layer_activations = encoder_output
            
            # Only use first 2 seconds worth of tokens (100 tokens at 50 Hz)
            # Slice: (batch, time_tokens, features) -> (batch, 100, features)
            layer_activations = layer_activations[:, :self.n_time_tokens, :]
            
            # Flatten: (batch, 100, features) -> (batch, 100*features)
            activations = layer_activations.flatten(start_dim=1)
            activations = activations.detach()
            
            # Clear activations for next forward pass
            self.encoder_activations.clear()
        
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
        mel, labels = batch
        logits = self.forward(mel) 
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
        mel, labels = batch
        logits = self.forward(mel) 
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
            # Resample to 16kHz for Whisper (assuming input is 20kHz)
            signal = self.resample_audio(signal.unsqueeze(0)).squeeze(0)
            signals.append(signal)
        
        # Stack signals: (batch, time)
        signals = torch.cat(signals, dim=0)  # (batch, time)
        # Process batch through Whisper preprocessing
        # Pad or trim to 30 seconds (480000 samples at 16kHz)
        signals = whisper.pad_or_trim(signals)
        
        # Convert to mel spectrogram (batch processing)
        mel = whisper.log_mel_spectrogram(signals, n_mels=self.n_mels)
        
        return mel, labels  

    def eval_collate_fn(self, batch):
        audio, targets = batch[0] # unbox wrapper added by dataloader 
        # convert labels to torch tensors 
        labels = {}
        if isinstance(targets, dict):
            for task_key, task_labels in targets.items():
                labels[task_key] = torch.from_numpy(task_labels)
        else:
            labels = torch.from_numpy(targets) 

        # resample audio to 16kHz    
        audio = self.resample_audio(audio)
        # Process batch through Whisper preprocessing
        # Pad or trim to 30 seconds (480000 samples at 16kHz)
        audio = whisper.pad_or_trim(audio)
        # Convert to mel spectrogram (batch processing)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        
        return mel, labels  

    
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
    