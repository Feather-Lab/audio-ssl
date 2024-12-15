from zipfile import ZipFile
import random
import torch

import torchvision
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

import numpy as np
from typing import List
import matplotlib.pyplot as plt

####################
# Set constants
####################
imagenet_normalize_mean = [0.485, 0.456, 0.406]
imagenet_normalize_std = [0.229, 0.224, 0.225]
imagenet_100_path = '/mnt/ceph/users/tyerxa/datasets/imagenet_100/'

class DeNormalizer(object):
    def __init__(self, mean=imagenet_normalize_mean, std=imagenet_normalize_std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img):
        return img * self.std + self.mean
    
imagenet_denormalizer = DeNormalizer(imagenet_normalize_mean, imagenet_normalize_std)

class Zip_ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, zip_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_path = zip_path
        self.zip_archvive = None

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        if self.zip_archvive is None:
            self.zip_archvive = ZipFile(self.zip_path)

        path_split = path.split("/")
        fh = self.zip_archvive.open(
            path_split[-3] + "/" + path_split[-2] + "/" + path_split[-1]
        )

        image = Image.open(fh)
        sample = image.convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            sample = self.target_transform(sample)

        return sample, target


class ImageNetValTransform:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, x):
        return self.transform(x)


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GaussianBlur(object):
    def __init__(self, p, min_sigma=0.1, max_sigma=2.0):
        self.p = p
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * (self.max_sigma - self.min_sigma) + self.min_sigma
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Barlow_Transform:
    def __init__(self, include_rotations=False, rotation_max_degree=0):

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)

        return y1, y2
    