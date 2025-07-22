import sys 
sys.path.append('byol-a')
from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020
from easydict import EasyDict
import torchaudio 
import torch
import lightning as L


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
        
        self.normalizer = PrecomputedNorm(self.stats)

        # Load pretrained weights.
        self.model = AudioNTT2020(d=self.config.feature_d)
        self.model.load_weight('byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', self.device)
        self.model = self.model.eval()
        # Need to manually freeze params here 
        self.model.trainable = False
        for name, param in self.model.named_parameters():
            param.requires_grad = False 

        self.proj_out_dim = 2048

    def forward(self, x):
        with torch.no_grad():
            x = self.normalizer((self.to_melspec(x) + torch.finfo(torch.float).eps).log())
            activations = self.model(x)
        return activations