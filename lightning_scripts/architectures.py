
import torch 
from torch import nn  
from torchvision.models.resnet import resnet50
from robustness.audio_models import resnet50 as resnet50_robusntess
class SSLAudioModel(nn.Module):
    def __init__(self, projector_dims=[512, 512], proj_out_dim=2048, n_classes=794, supervised=False, **kwargs):
        super().__init__()
        self.supervised = supervised

        self.f = resnet50()
        self.f.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.f.fc = nn.Identity()

        # projection head (Following exactly barlow twins offical repo)
        projector_dims = [proj_out_dim] + projector_dims
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=False)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=False))
        self.g = nn.Sequential(*layers)
        if supervised:
            self.lin_cls = nn.Linear(proj_out_dim, n_classes)

    def forward(self, x):
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out = self.g(feature)
        if not self.supervised:
            return feature, out, None 
        else:
            logits = self.lin_cls(feature)
        return feature, out, logits
    
    
class SSLAudioModelWMetamers(nn.Module):
    def __init__(self, projector_dims=[512, 512], proj_out_dim=2048, n_classes=794, supervised=False, **kwargs):
        super().__init__()
        self.supervised = supervised

        self.f = resnet50_robusntess()
        self.f.fc = nn.Identity()

        # projection head (Following exactly barlow twins offical repo)
        projector_dims = [proj_out_dim] + projector_dims
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=False)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=False))
        self.g = nn.Sequential(*layers)
        if supervised:
            self.lin_cls = nn.Linear(proj_out_dim, n_classes)

    def forward(self, x):
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out = self.g(feature)
        if not self.supervised:
            return feature, out, None 
        else:
            logits = self.lin_cls(feature)
        return feature, out, logits