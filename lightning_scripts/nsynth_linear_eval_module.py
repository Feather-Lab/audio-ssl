"""
NSynth Linear Evaluation Module.

Trains a linear classifier on top of frozen SSL feature extractors for NSynth tasks.
"""

import torch
import torch.nn as nn
import lightning as L
from typing import Union
from lightning_ssl_classifier import SSLClassifier
from audio_ssl.misc import LARS, CosineWarmupScheduler
from torchmetrics.classification import Accuracy
import torch.nn.functional as F


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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.layer_out = layer_out
        self.num_classes = num_classes
        self.w_mlp = w_mlp
        self.mlp_dim = mlp_dim
        self.with_dropout = with_dropout
        
        # Create a dummy config for SSLClassifier that matches our needs
        # We'll override the classifier head
        ssl_config = config.copy()
        ssl_config['model']['arch_kwargs']['num_classes'] = num_classes  # Single task
        
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
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension based on layer_out and architecture."""
        backbone = self.config['model']['arch_kwargs'].get('backbone', '')
        time_avg_rep = self.config['model']['arch_kwargs'].get('time_average', True)
        no_avgpool = self.config['model']['arch_kwargs'].get('no_avgpool', False)
        
        if backbone == 'kell2018' or 'kell2018' in str(backbone):
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
        # Use SSLClassifier's forward to get features
        # We need to extract features, not the full forward pass
        with torch.no_grad():
            if self.feature_extractor_wrapper.byola_arch:
                _, _, _ = self.feature_extractor_wrapper.feature_extractor.model(
                    x, with_latent=False, fake_relu=False
                )
                activations = self.feature_extractor_wrapper.act_hook_dict[self.layer_out]
            else:
                predictions, rep, all_outputs = self.feature_extractor_wrapper.feature_extractor.model(
                    x, with_latent=True, fake_relu=False
                )
                activations = all_outputs[self.layer_out]
            
            # Time average and flatten
            if self.feature_extractor_wrapper.time_avg_rep:
                activations = activations.mean(dim=-1).view(activations.shape[0], -1)
            else:
                activations = activations.view(activations.shape[0], -1)
            
            activations = activations.detach()
        
        # Apply dropout
        if self.dropout:
            activations = self.dropout(activations)
        
        # Apply MLP if present
        if self.mlp:
            activations = self.mlp(activations)
        
        # Classifier
        logits = self.classifier(activations)
        
        return logits
    
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

