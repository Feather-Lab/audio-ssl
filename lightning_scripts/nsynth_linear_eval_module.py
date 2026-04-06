"""
NSynth Linear Evaluation Module.

Trains a linear classifier on top of frozen SSL feature extractors for NSynth tasks.
"""

import torch
import torch.nn as nn
import lightning as L
import whisper
import torchaudio
import sys
from typing import Union, Optional
from lightning_ssl_classifier import SSLClassifier
from audio_ssl.misc import LARS, CosineWarmupScheduler
from torchmetrics.classification import Accuracy
import torch.nn.functional as F
from nsynth_dataset import NsynthDataset

from byola_lightning_module import BYOLAModule
from audiomae_encoder_utils import (
    AUDIOMAE_DIM,
    AUDIOMAE_FREQ_PATCHES,
    AUDIOMAE_SR,
    AUDIOMAE_TIME_PATCHES,
    preprocess_waveform as audiomae_preprocess_waveform,
)


class NSynthLinearEvalModule(L.LightningModule):
    """
    Linear evaluation module for NSynth tasks.
    
    Wraps SSLClassifier for feature extraction and adds a single-task linear head.
    """
    
    def __init__(
        self,
        config,
        ckpt_path,
        layer_out,
        num_classes,
        supervised_backbone=False,
        w_mlp=False,
        mlp_dim=512,
        with_dropout=False,
        nsynth_root: str = "/mnt/home/igriffith/ceph/datasets/nsynth",
        task: str = "family",
        label_field: Optional[str] = None,
        duration: Optional[float] = None,
        fade_window_duration: float = 0.01,
        sample_rate: int = 20000,
        use_whisper: bool = False,
        use_byola: bool = False,
        use_audiomae: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.layer_out = layer_out
        self.num_classes = num_classes
        self.w_mlp = w_mlp
        self.mlp_dim = mlp_dim
        self.with_dropout = with_dropout
        self.nsynth_root = nsynth_root
        self.task = task
        self.label_field = label_field
        self.duration = duration
        self.fade_window_duration = fade_window_duration
        self.sample_rate = sample_rate
        
        # Track encoder type
        if 'model' not in self.config:
            self.config['model'] = {}
        if 'arch_kwargs' not in self.config['model']:
            self.config['model']['arch_kwargs'] = {}
        self.use_whisper = use_whisper or bool(self.config['model'].get('whisper_model'))
        self.use_byola = use_byola or bool(self.config['model'].get('byola_arch'))
        self.use_audiomae = use_audiomae
        
        if self.use_audiomae:
            self._init_audiomae_encoder()
            feature_dim = self.audiomae_feature_dim
            self.feature_extractor_wrapper = None
        elif self.use_whisper:
            self._init_whisper_encoder()
            feature_dim = self.classifier_input_dim
            self.feature_extractor_wrapper = None
        elif self.use_byola:
            # Use BYOLAModule directly (same as speech_commands)
            # Initialize on CPU first, Lightning will move to GPU
            self.byola_module = BYOLAModule(config=config)
            # Move to device if available (Lightning will handle this, but do it here for safety)
            if torch.cuda.is_available():
                self.byola_module = self.byola_module.cuda()
            feature_dim = 2048  # BYOL-A final output dimension
            self.feature_extractor_wrapper = None
        else:
            # Create a config for SSLClassifier that matches our needs
            # We'll override the classifier head
            import copy
            ssl_config = copy.deepcopy(config)  # Deep copy to avoid modifying original config
            # Ensure arch_kwargs exists (may have been converted from arch_params in eval_nsynth_linear.py)
            if 'arch_kwargs' not in ssl_config['model']:
                ssl_config['model']['arch_kwargs'] = {}
            ssl_config['model']['arch_kwargs']['num_classes'] = num_classes  # Single task
            # For supervised models, ensure arch_params is preserved for checkpoint loading
            # (arch_kwargs is used for SSLClassifier internal logic, arch_params for model loading)
            if supervised_backbone and 'arch_params' not in ssl_config['model'] and 'arch_kwargs' in ssl_config['model']:
                # Restore arch_params from arch_kwargs for supervised model loading
                ssl_config['model']['arch_params'] = copy.deepcopy(ssl_config['model']['arch_kwargs'])
            
            # Initialize SSLClassifier for feature extraction
            self.feature_extractor_wrapper = SSLClassifier(
                config=ssl_config,
                ckpt_path=ckpt_path,
                layer_out=layer_out,
                supervised_backbone=supervised_backbone
            )
            
            # Get feature dimension from the wrapper's layer size dict logic
            feature_dim = self._get_feature_dim()
        
        # Build classifier head
        self.mlp = None
        if w_mlp:
            # MLP head (following SSLClassifier pattern)
            hidden_dims = [feature_dim, mlp_dim]
            layers = []
            for i in range(len(hidden_dims) - 1):
                layers.append(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False)
                )
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                layers.append(nn.ReLU())
            self.mlp = nn.Sequential(*layers)
            feature_dim = mlp_dim
        
        # Linear classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Dropout
        if with_dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
    
    def _init_audiomae_encoder(self):
        """Initialize frozen AudioMAE encoder with hook on specified layer."""
        from transformers import AutoModel
        from transformers.modeling_utils import PreTrainedModel

        _orig_mark = PreTrainedModel.mark_tied_weights_as_initialized
        def _safe_mark(self_model, loading_info):
            if not hasattr(self_model, "all_tied_weights_keys"):
                self_model.all_tied_weights_keys = {}
            return _orig_mark(self_model, loading_info)
        PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark

        wrapper = AutoModel.from_pretrained(
            "hance-ai/audiomae", trust_remote_code=True
        )
        self.audiomae_encoder = wrapper.encoder
        for param in self.audiomae_encoder.parameters():
            param.requires_grad = False
        self.audiomae_encoder.eval()

        encoder_layer = self.config['model']['arch_kwargs'].get('encoder_layer', 11)
        self.audiomae_layer_idx = int(encoder_layer)
        self.audiomae_n_blocks = len(self.audiomae_encoder.blocks)
        self.audiomae_use_norm = self.audiomae_layer_idx >= self.audiomae_n_blocks

        self.audiomae_activations = {}
        if not self.audiomae_use_norm:
            def hook_fn(_module, _input, output):
                out = output[0] if isinstance(output, tuple) else output
                self.audiomae_activations["layer_output"] = out.detach()
            if self.audiomae_layer_idx < 0 or self.audiomae_layer_idx >= self.audiomae_n_blocks:
                raise ValueError(
                    f"AudioMAE encoder_layer {self.audiomae_layer_idx} out of range "
                    f"(0-{self.audiomae_n_blocks - 1})"
                )
            self.audiomae_encoder.blocks[self.audiomae_layer_idx].register_forward_hook(hook_fn)

        time_pool = self.config['model']['arch_kwargs'].get('time_average', True)
        self.audiomae_time_pool = time_pool
        if time_pool:
            self.audiomae_feature_dim = AUDIOMAE_DIM * AUDIOMAE_FREQ_PATCHES  # 6144
        else:
            self.audiomae_feature_dim = AUDIOMAE_DIM * AUDIOMAE_FREQ_PATCHES * AUDIOMAE_TIME_PATCHES

    def _extract_audiomae_features(self, x):
        """Extract features from frozen AudioMAE encoder."""
        if x.dim() == 3:
            x = x.squeeze(1)

        mel = audiomae_preprocess_waveform(x, sr=self.sample_rate).to(x.device)

        with torch.no_grad():
            self.audiomae_activations.clear()
            full_out = self.audiomae_encoder.forward_features(mel)

            if self.audiomae_use_norm:
                tokens = full_out
            else:
                tokens = self.audiomae_activations["layer_output"]

            tokens = tokens[:, 1:, :]  # remove CLS
            feats = tokens.reshape(
                tokens.shape[0],
                AUDIOMAE_FREQ_PATCHES,
                AUDIOMAE_TIME_PATCHES,
                AUDIOMAE_DIM,
            ).permute(0, 3, 1, 2)  # (B, D, freq, time)

            if self.audiomae_time_pool:
                feats = feats.mean(dim=-1)

            activations = feats.flatten(start_dim=1).detach()
            self.audiomae_activations.clear()
        return activations

    def _init_whisper_encoder(self):
        """Initialize Whisper encoder components."""
        whisper_model_name = self.config['model'].get('whisper_model', 'large-v3-turbo')
        whisper_model = whisper.load_model(whisper_model_name)
        self.n_mels = whisper_model.dims.n_mels
        self.whisper_encoder = whisper_model.encoder
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        self.whisper_encoder.eval()
        
        # Determine encoder layer index and feature dimensions
        encoder_layer = self.config['model']['arch_kwargs'].get('encoder_layer', 31)
        if isinstance(encoder_layer, str):
            try:
                encoder_layer = int(encoder_layer)
            except ValueError:
                raise ValueError(f"encoder_layer must be integer-like, got {encoder_layer}")
        self.encoder_layer_idx = int(encoder_layer)
        self.n_time_tokens = self.config['model'].get('whisper_n_time_tokens', 100)
        n_audio_state = whisper_model.dims.n_audio_state
        self.classifier_input_dim = self.n_time_tokens * n_audio_state
        self.encoder_activations = {}
        self.whisper_target_sample_rate = 16000
        self._register_encoder_hook()
    
    def _register_encoder_hook(self):
        """Register hook on Whisper encoder layer to capture activations."""
        def hook_fn(module, inputs, outputs):
            if isinstance(outputs, tuple):
                self.encoder_activations['layer_output'] = outputs[0].detach()
            else:
                self.encoder_activations['layer_output'] = outputs.detach()
        
        encoder_layers = self.whisper_encoder.blocks
        num_layers = len(encoder_layers)
        if self.encoder_layer_idx < 0 or self.encoder_layer_idx >= num_layers:
            raise ValueError(
                f"Encoder layer index {self.encoder_layer_idx} out of range (0-{num_layers-1})"
            )
        encoder_layers[self.encoder_layer_idx].register_forward_hook(hook_fn)
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension based on layer_out and architecture."""
        # Handle both arch_kwargs and arch_params (for supervised models)
        arch_config = self.config['model'].get('arch_kwargs', {})
        if not arch_config and 'arch_params' in self.config['model']:
            arch_config = self.config['model']['arch_params']
        
        # Try to get backbone from arch_kwargs, or infer from arch_name
        backbone = arch_config.get('backbone', '')
        if not backbone:
            arch_name = self.config['model'].get('arch_name', '')
            if 'kell2018' in arch_name:
                backbone = 'kell2018'
            elif 'resnet18' in arch_name or 'resnet_multi_task18' in arch_name:
                backbone = 'resnet18'
            elif 'resnet' in arch_name:
                backbone = 'resnet50'
        
        time_avg_rep = arch_config.get('time_average', True)
        no_avgpool = arch_config.get('no_avgpool', False)
        
        # Check for BYOL-A (AudioNTT2020)
        if backbone == 'AudioNTT2020' or 'byol' in str(backbone).lower() or 'byola' in str(backbone).lower():
            # BYOL-A layer sizes (from SSLClassifier)
            if time_avg_rep:
                layer_size_dict = {
                    'input_after_preproc': 211,
                    'relu0': 5256960,
                    'relu1': 1310400,
                    'relu2': 323584,
                    'relu3': 98304,
                    'relu4': 98304,
                    'final': 2048
                }
            else:
                # For non-time-averaged, use the same sizes (time dimension is flattened)
                layer_size_dict = {
                    'input_after_preproc': 211,
                    'relu0': 5256960,
                    'relu1': 1310400,
                    'relu2': 323584,
                    'relu3': 98304,
                    'relu4': 98304,
                    'final': 2048
                }
        elif backbone == 'kell2018' or 'kell2018' in str(backbone):
            if time_avg_rep:
                layer_size_dict = {
                    'input_after_preproc': 211,
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
                    'relufc': 4096
                }
            else:
                layer_size_dict = {
                    'input_after_preproc': 82290,
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
                    'relufc': 4096
                }
        elif backbone in ['resnet18', 'resnet_multi_task18']:
            if time_avg_rep:
                layer_size_dict = {
                    'input_after_preproc': 211,
                    'conv1': 6784,
                    'bn1': 6784,
                    'conv1_relu1': 6784,
                    'maxpool1': 3392,
                    'layer1': 3392,
                    'layer2': 3456,
                    'layer3': 3584,
                    'layer4': 3584,
                    'avgpool': 512,
                    'final': 512
                }
            else:
                layer_size_dict = {
                    'input_after_preproc': 82290,
                    'conv1': 1322880,
                    'bn1': 1322880,
                    'conv1_relu1': 1322880,
                    'maxpool1': 332416,
                    'layer1': 332416,
                    'layer2': 169344,
                    'layer3': 89600,
                    'layer4': 46592,
                    'avgpool': 512,
                    'final': 512
                }
            if no_avgpool:
                layer_size_dict['avgpool'] = layer_size_dict['layer4']
        else:  # resnet50
            if time_avg_rep:
                layer_size_dict = {
                    'input_after_preproc': 211,
                    'conv1': 6784,
                    'bn1': 6784,
                    'conv1_relu1': 6784,
                    'maxpool1': 3392,
                    'layer1': 3392,
                    'layer2': 3456,
                    'layer3': 3584,
                    'layer4': 3584,
                    'avgpool': 2048,
                    'final': 2048
                }
            else:
                layer_size_dict = {
                    'input_after_preproc': 82290,
                    'conv1': 1322880,
                    'bn1': 1322880,
                    'conv1_relu1': 1322880,
                    'maxpool1': 332416,
                    'layer1': 332416,
                    'layer2': 169344,
                    'layer3': 89600,
                    'layer4': 186368,
                    'avgpool': 2048,
                    'final': 2048
                }
                if no_avgpool:
                    layer_size_dict['avgpool'] = layer_size_dict['layer4']
        
        if self.layer_out not in layer_size_dict:
            raise ValueError(
                f"Layer '{self.layer_out}' not found in layer_size_dict. "
                f"Available layers: {list(layer_size_dict.keys())}"
            )
        
        return layer_size_dict[self.layer_out]
    
    def forward(self, x):
        """
        Forward pass through feature extractor and classifier.
        
        Args:
            x: Audio tensor of shape (batch, 1, time) or (batch, time)
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
        """
        if self.use_audiomae:
            activations = self._extract_audiomae_features(x)
        elif self.use_whisper:
            activations = self._extract_whisper_features(x)
        else:
            activations = self._extract_ssl_features(x)
        
        # Apply dropout
        if self.dropout:
            activations = self.dropout(activations)
        
        # Apply MLP if present
        if self.mlp:
            activations = self.mlp(activations)
        
        # Classifier
        logits = self.classifier(activations)
        
        return logits
    
    def _extract_ssl_features(self, x):
        """Extract features using SSL backbone."""
        if self.use_byola:
            # Use BYOLAModule directly (same as speech_commands)
            # Input x is (batch, 1, time) or (batch, time) - raw audio
            if x.dim() == 3:
                x = x.squeeze(1)  # (batch, 1, time) -> (batch, time)
            activations = self.byola_module(x)
            # BYOL-A outputs are already time-averaged to [batch, 2048]
            return activations
        
        # Use SSLClassifier's forward to get features
        with torch.no_grad():
            if self.feature_extractor_wrapper.byola_arch:
                _, _, _ = self.feature_extractor_wrapper.feature_extractor.model(
                    x, with_latent=False, fake_relu=False
                )
                activations = self.feature_extractor_wrapper.act_hook_dict[self.layer_out]
            else:
                _, _, all_outputs = self.feature_extractor_wrapper.feature_extractor.model(
                    x, with_latent=True, fake_relu=False
                )
                activations = all_outputs[self.layer_out]
            
            # Time average and flatten
            if self.feature_extractor_wrapper.time_avg_rep:
                activations = activations.mean(dim=-1).view(activations.shape[0], -1)
            else:
                activations = activations.view(activations.shape[0], -1)
            
            activations = activations.detach()
        return activations
    
    def _extract_whisper_features(self, x):
        """Extract features from Whisper encoder."""
        if x.dim() == 3:
            x = x.squeeze(1)
        original_device = x.device
        audio = x.detach().float().cpu()
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels).to(original_device)
        
        with torch.no_grad():
            encoder_output = self.whisper_encoder(mel)
            if 'layer_output' in self.encoder_activations:
                layer_activations = self.encoder_activations['layer_output']
            else:
                layer_activations = encoder_output
            layer_activations = layer_activations[:, :self.n_time_tokens, :]
            activations = layer_activations.flatten(start_dim=1)
            self.encoder_activations.clear()
        return activations
    
    def _step(self, batch, batch_idx, step_type):
        """Common step for train/val/test."""
        audio, labels = batch
        
        # Ensure audio has channel dimension
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (batch, time) -> (batch, 1, time)
        
        logits = self.forward(audio)
        loss = self.criterion(logits, labels)
        
        # Compute accuracy
        if step_type == 'train':
            acc = self.train_accuracy(logits, labels)
        elif step_type == 'val':
            acc = self.val_accuracy(logits, labels)
        else:
            acc = self.test_accuracy(logits, labels)
        
        # Log metrics
        self.log(f"{step_type}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{step_type}_acc", acc, prog_bar=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
    
    def predict_step(self, batch):
        """Predict step for final evaluation."""
        audio, labels = batch
        
        # Ensure audio has channel dimension
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        logits = self.forward(audio)
        
        # Compute top-1 and top-5 accuracy
        top1 = (logits.argmax(dim=-1) == labels).float().mean()
        
        # Top-5 accuracy
        top5_preds = logits.topk(k=min(5, self.num_classes), dim=-1).indices
        top5 = (top5_preds == labels.unsqueeze(-1)).any(dim=-1).float().mean()
        
        return {
            'top1': top1,
            'top5': top5,
            'logits': logits,
            'labels': labels
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.config['hparas']['optimizer'] == "LARS":
            lr = self.config['hparas']['lr'] * self.config['hparas'].get('global_batch_size', 256) / 256
            self.optimizer = LARS(
                self.classifier.parameters() if not self.mlp else list(self.mlp.parameters()) + list(self.classifier.parameters()),
                lr=lr,
                weight_decay=1e-6,
                momentum=0.9,
                weight_decay_filter=True,
                lars_adaptation_filter=True,
            )
        else:
            lr = self.config['hparas']['lr']
            opt = getattr(torch.optim, self.config['hparas']['optimizer'])
            params = self.classifier.parameters() if not self.mlp else list(self.mlp.parameters()) + list(self.classifier.parameters())
            self.optimizer = opt(params, lr=lr)
        
        # Learning rate scheduler
        if self.config['hparas'].get('lr_schedule', False):
            total_training_steps = self.total_training_steps()
            num_warmup_steps = self.compute_warmup(
                total_training_steps,
                self.config['hparas'].get('num_warmup_steps_or_ratio', 0)
            )
            lr_scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                batch_size=self.config['hparas'].get('global_batch_size', 256),
                warmup_steps=num_warmup_steps,
                max_steps=total_training_steps,
                lr=lr
            )
            return [self.optimizer], [{
                'scheduler': lr_scheduler,
                'interval': 'step',
            }]
        
        return [self.optimizer]
    
    def total_training_steps(self) -> int:
        """Compute total training steps."""
        # This will be set by the trainer, but we need a fallback
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            return self.trainer.estimated_stepping_batches
        # Fallback calculation
        dataset_size = len(self.trainer.train_dataloader.dataset)
        num_devices = self.trainer.num_devices
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        return int(max_estimated_steps)
    
    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        """Compute warmup steps."""
        return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_warmup_steps
    
    def train_dataloader(self):
        """Create training dataloader."""
        train_dataset = NsynthDataset(
            nsynth_root=self.nsynth_root,
            split='train',
            task=self.task,
            label_field=self.label_field if self.task == 'other' else None,
            sample_rate=self.sample_rate,
            duration=self.duration,
            fade_window_duration=self.fade_window_duration,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['hparas']['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        val_dataset = NsynthDataset(
            nsynth_root=self.nsynth_root,
            split='valid',
            task=self.task,
            label_field=self.label_field if self.task == 'other' else None,
            sample_rate=self.sample_rate,
            duration=self.duration,
            fade_window_duration=self.fade_window_duration,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['hparas']['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
        )

