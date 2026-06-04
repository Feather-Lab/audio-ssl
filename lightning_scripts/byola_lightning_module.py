import sys 
sys.path.append('byol-a')
from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020
from easydict import EasyDict
import torchaudio 
import torch
import lightning as L
from default_paths import WORKING_DIRECTORY


class BYOLAModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        config = EasyDict(config)
        self.config = config

        # self.stats = [10158236.,  51190964.] ## Stats of jsinV3 - can use if needeing to do inference
        self.stats = [-5.4919195,  5.0389895]

        self.to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        )      

                # Layer selection for classifier attachment
        self.layer_name = config.get('classifier_layer', None)  # e.g., 'layer3', 'avgpool', None for final
        self.layer_output = None
        
        self.normalizer = PrecomputedNorm(self.stats)

        # Load pretrained weights.
        self.model = AudioNTT2020(d=self.config.feature_d)
        # Determine device - use CPU for loading, Lightning will move to GPU later
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weight_path = WORKING_DIRECTORY / "byol-a" / "pretrained_weights" / "AudioNTT2020-BYOLA-64x96d2048.pth"
        self.model.load_weight(str(weight_path), device)
        self.model = self.model.eval()
        # Need to manually freeze params here 
        self.model.trainable = False
        for name, param in self.model.named_parameters():
            param.requires_grad = False 

        if self.layer_name:
            self._register_hook()
            self.proj_out_dim = self._get_layer_output_dim()
        else:
            self.proj_out_dim = 2048

            
    def _register_hook(self):
        """Register forward hook to capture intermediate layer outputs"""
        def hook_fn(module, input, output):
            self.layer_output = output
        
        # Parse layer specification (e.g., 'features.2' or 'features.6')
        parts = self.layer_name.split('.')
        
        # Navigate to the target module
        target_module = self.model
        for part in parts:
            if part.isdigit():
                # Access Sequential layer by index
                target_module = target_module[int(part)]
            else:
                # Access by attribute name
                target_module = getattr(target_module, part)
        
        target_module.register_forward_hook(hook_fn)

    def _get_layer_output_dim(self):
        """Probe the layer to get its output dimension"""
        with torch.no_grad():
             # Typical mel-spectrogram shape for BYOLA for 2 second audio at 16kHz
            dummy_input = torch.randn(1, 1, 64, 201).to(self.device) 
            _ = self.model(dummy_input)
            
            if self.layer_output is None:
                raise ValueError(f"Layer '{self.layer_name}' did not produce output. Check layer name.")
            
            # Handle different output shapes
            output_dim = self.layer_output.flatten(start_dim=1).shape[-1]
            self.layer_output = None  # Reset
            return output_dim

    def forward(self, x):
        with torch.no_grad():
            # Convert audio to mel spectrogram: (batch, time) -> (batch, mel, time)
            mel = self.to_melspec(x)
            # Normalize: (batch, mel, time)
            mel_norm = self.normalizer((mel + torch.finfo(torch.float).eps).log())
            # Add channel dimension: (batch, mel, time) -> (batch, 1, mel, time)
            mel_norm = mel_norm.unsqueeze(1)
            # Forward through model: (batch, 1, mel, time) -> (batch, time, d) -> (batch, d) after pooling
            if self.layer_name:
                # Run through feature extractor to populate layer_output via hook
                _ = self.model(mel_norm)
                activations = self.layer_output.detach()
                self.layer_output = None  # Reset for next forward pass
                
                # Flatten if needed
                if len(activations.shape) > 2:
                    activations = activations.reshape(activations.shape[0], -1)
            else:
                activations = self.model(mel_norm)
        return activations