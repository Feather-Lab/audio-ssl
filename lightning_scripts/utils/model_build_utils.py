
from lightning_scripts.lightning_classifier_matched_speech_in_noise import (
    LitWordAudioSetModel,
)
from pathlib import Path
import os
from default_paths import MODEL_CHECKPOINT_DIR
from lightning_scripts.lightning_ssl_matched_speech_in_noise import LitAudioSSL
import torch.nn as nn
import yaml

exp_dir = MODEL_CHECKPOINT_DIR

def get_checkpoint_path(config_path: Path, exp_dir: Path = exp_dir) -> str:  
    checkpoint_dir = exp_dir / f"{config_path.stem}/checkpoints"
    all_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    
    # First try to find checkpoints with "best_val" in the name
    best_val_checkpoints = [ckpt for ckpt in all_checkpoints if "best_val" in ckpt.name]
    
    if best_val_checkpoints:
        # Return the latest best_val checkpoint
        checkpoint_path = str(sorted(best_val_checkpoints, key=os.path.getctime)[-1])
    else:
        # Fall back to latest checkpoint overall
        checkpoint_path = str(sorted(all_checkpoints, key=os.path.getctime)[-1])
    
    return checkpoint_path


class FeatureExtractor(nn.Module):
    """Feature extractor for SSL models (has .f attribute) with optional hooks into intermediate layers."""
    def __init__(self, feature_extractor, layer_out=None):
        super(FeatureExtractor, self).__init__()
        # Keep the full lightning feature extractor so we can optionally call with_latent=True
        self.feature_extractor = feature_extractor.eval()
        self.front_end = self.feature_extractor.front_end
        self.backbone = self.feature_extractor.model.f.eval()
        self.layer_out = layer_out
        # freeze the front end 
        for param in self.front_end.parameters():
            param.requires_grad = False

    def forward(self, x):
        # When no specific layer is requested, preserve the previous behaviour:
        # run the front end then the backbone .f module.
        if self.layer_out is None:
            x, _ = self.front_end(x, None)
            return self.backbone(x)

        # Otherwise, leverage the model's latent output dictionary.
        _, _, all_outputs = self.feature_extractor(x, with_latent=True, fake_relu=False)
        if self.layer_out not in all_outputs:
            raise KeyError(f"Requested layer '{self.layer_out}' not found in model outputs.")
        return all_outputs[self.layer_out]


class SupervisedFeatureExtractor(nn.Module):
    """Feature extractor for supervised models (uses .model and forward hooks)"""
    def __init__(self, feature_extractor, layer_out='relufc'):
        super(SupervisedFeatureExtractor, self).__init__()
        self.front_end = feature_extractor.front_end
        self.model = feature_extractor.model.eval()
        # freeze the front end 
        for param in self.front_end.parameters():
            param.requires_grad = False
            
        self.layer_out = layer_out
    def forward(self, x):
        # Clear previous output
        self.relufc_output = None
        
        # Pass through front end
        x, _ = self.front_end(x, None)
        
        # Forward through model with with_latent=True to trigger hooks
        # The hook will capture relufc output
        _, _, all_outputs = self.model(x, with_latent=True, fake_relu=False)
        
        # Return relufc output (either from hook or all_outputs)
        return all_outputs[self.layer_out]


def get_model(config_path, supervised=False, layer_out="relu4", random=False):
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    checkpoint_path = get_checkpoint_path(config_path)
    if random:
        module = LitAudioSSL(config)
        model = FeatureExtractor(module.model, layer_out=layer_out).eval().cuda()
        return model
    if supervised:
        module = LitWordAudioSetModel.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, strict=True).eval()
        model = SupervisedFeatureExtractor(module.model, layer_out=layer_out).eval().cuda()
    else:
        module = LitAudioSSL.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, strict=True).eval()
        model = FeatureExtractor(module.model, layer_out=layer_out).eval().cuda()

    return model