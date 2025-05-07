
import torch 
from torch import nn  
from torchvision.models.resnet import resnet50
from robustness.audio_models import resnet50 as resnet50_robusntess
import robustness.audio_models as robustness_architectures
import einops 

class ProjectionHead(nn.Module):
    def __init__(self, projector_dims):
        super(ProjectionHead, self).__init__()
        self.projector_dims = projector_dims
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=False)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=False))
        self.g = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.g(x)

class SSLBaseModel(nn.Module):
    def __init__(self, backbone='resnet50', projector_dims=[512, 512], proj_out_dim=2048, in_channels=1, num_classes=794, supervised=False, **kwargs):
        super().__init__()
        self.supervised = supervised
        self.backbone = backbone

        self.f = robustness_architectures.__dict__[backbone]()
        self.f.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.f.fc = nn.Identity()
        if "max_to_avg_pool" in kwargs:
            self.f.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

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
            self.lin_cls = nn.Linear(proj_out_dim, num_classes)

    def forward(self, x):
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out = self.g(feature)
        if not self.supervised:
            return feature, out, None 
        else:
            logits = self.lin_cls(feature.detach())
        return feature, out, logits


class SSLBaseModelSingleTask(nn.Module):
    def __init__(self, backbone='resnet50', projector_dims=[512, 512], proj_out_dim=2048, in_channels=1, num_classes=794, supervised=False, frame_wise_loss=False, **kwargs):
        super().__init__()
        self.supervised = supervised
        self.backbone = backbone
        self.frame_wise_loss = frame_wise_loss

        self.f = robustness_architectures.__dict__[backbone]()
        
        if 'resnet' in backbone:
            self.f.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.f.fc = nn.Identity()
            if "max_to_avg_pool" in kwargs:
                self.f.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            if self.frame_wise_loss:
                # don't average over time dimensions - will fold into batch dimension 
                self.freq_avg = nn.AdaptiveAvgPool2d((1, None))
                # self.f.avgpool = nn.AdaptiveAvgPool2d((1, None))
                self.ssl_rep_name = 'layer4'
            if "no_avgpool" in kwargs:
                self.f.avgpool = nn.Flatten()
                proj_out_dim = 46592

        elif 'kell' in backbone:
            self.f.dropout = nn.Identity()
            self.f.classification = nn.Identity()
        

        # projection head (Following exactly barlow twins offical repo)
        ## Assumes same dims for inv and equi tasks 
        projector_dims = [proj_out_dim] + projector_dims
        self.g_inv = ProjectionHead(projector_dims)
        if supervised:
            if isinstance(num_classes, dict): # Make multiple fully conected layers
                all_fc_layers = {}
                for task in num_classes.keys():
                    all_fc_layers[task] = nn.Linear(proj_out_dim, num_classes[task]) 
                self.lin_cls = nn.ModuleDict(all_fc_layers)
            else:
                self.lin_cls = nn.Linear(proj_out_dim, num_classes)

    def forward(self, x):
        if self.frame_wise_loss:
            _, feature, all_outputs = self.f(x, with_latent=True)
            feature = torch.flatten(feature, start_dim=1)
            ## Fold time into batch dimension - time is collated 
            ssl_feature = all_outputs[self.ssl_rep_name]
            ssl_feature = self.freq_avg(ssl_feature)
            ssl_feature = einops.rearrange(ssl_feature, "b c f t -> (b t) (c f)")
            inv_out = self.g_inv(ssl_feature)
        else:
            x = self.f(x)
            feature = torch.flatten(x, start_dim=1)
            inv_out = self.g_inv(feature)
        if not self.supervised:
            return feature, inv_out, None 
        else:
            if isinstance(self.lin_cls, nn.ModuleDict): 
                logits = {}
                for task, fc_l in self.lin_cls.items():
                    logits[task] = fc_l(feature.detach())
            else:
                logits = self.lin_cls(feature.detach())
        return feature, inv_out, logits
        

class SSLBaseModelDualTask(nn.Module):
    def __init__(self, backbone='resnet50', projector_dims=[512, 512], proj_out_dim=2048, in_channels=1, num_classes=794, supervised=False, **kwargs):
        super().__init__()
        self.supervised = supervised
        self.backbone = backbone
        self.f = robustness_architectures.__dict__[backbone]()

        if 'resnet' in backbone:
            self.f.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.f.fc = nn.Identity()
            if "max_to_avg_pool" in kwargs:
                self.f.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            if self.frame_wise_loss:
                # don't average over time dimensions - will fold into batch dimension 
                self.freq_avg = nn.AdaptiveAvgPool2d((1, None))
                # self.f.avgpool = nn.AdaptiveAvgPool2d((1, None))
                self.ssl_rep_name = 'layer4'

        elif 'kell' in backbone:
            self.f.dropout = nn.Identity()
            self.f.classification = nn.Identity()
        # projection head (Following exactly barlow twins offical repo)
        ## Assumes same dims for inv and equi tasks 
        projector_dims = [proj_out_dim] + projector_dims
        self.g_inv = ProjectionHead(projector_dims)
        self.g_equi = ProjectionHead(projector_dims)
        if supervised:
            if isinstance(num_classes, dict): # Make multiple fully conected layers
                all_fc_layers = {}
                for task in num_classes.keys():
                    all_fc_layers[task] = nn.Linear(proj_out_dim, num_classes[task]) 
                self.lin_cls = nn.ModuleDict(all_fc_layers)
            else:
                self.lin_cls = nn.Linear(proj_out_dim, num_classes)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        inv_out = self.g_inv(feature)
        equi_out = self.g_equi(feature)
        if not self.supervised:
            return feature, (inv_out, equi_out), None 
        else:
            if isinstance(self.lin_cls, nn.ModuleDict): 
                logits = {}
                for task, fc_l in self.lin_cls.items():
                    logits[task] = fc_l(feature.detach())
            else:
                logits = self.lin_cls(feature.detach())
        return feature, (inv_out, equi_out), logits


class SSLAudioModel(nn.Module):
    def __init__(self, projector_dims=[512, 512], proj_out_dim=2048, num_classes=794, supervised=False, **kwargs):
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
            self.lin_cls = nn.Linear(proj_out_dim, num_classes)

    def forward(self, x):
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out = self.g(feature)
        if not self.supervised:
            return feature, out, None 
        else:
            logits = self.lin_cls(feature.detach())
        return feature, out, logits
    
    
class SSLAudioModelWMetamers(nn.Module):
    def __init__(self, projector_dims=[512, 512], proj_out_dim=2048, num_classes=794, supervised=False, **kwargs):
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
            self.lin_cls = nn.Linear(proj_out_dim, num_classes)

    def forward(self, x):
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out = self.g(feature)
        if not self.supervised:
            return feature, out, None 
        else:
            logits = self.lin_cls(feature.detach())
        return feature, out, logits