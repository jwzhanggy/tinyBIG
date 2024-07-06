# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import transforms, Compose, ToTensor, Normalize, Lambda
from torchvision.datasets import MNIST, CIFAR10, ImageNet

from tinybig.data.base_data import dataloader


class vision_dataloader(dataloader):
    def __init__(self, name='vision_dataloader', *args, **kwargs):
        super().__init__(name=name)

    @abstractmethod
    def load(self):
        pass


class imagenet(vision_dataloader):

    def __init__(self, name='imagenet', train_batch_size=64, test_batch_size=64):
        super().__init__(name=name)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    # @staticmethod
    # def flatten(x):
    #     x = torch.flatten(x)
    #     return x.view(-1)

    def load(self, cache_dir='./data/', *args, **kwargs):
        imagenet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torch.flatten
        ])

        train_loader = DataLoader(
            ImageNet(root=cache_dir, split='train', transform=imagenet_transform),
            batch_size=self.train_batch_size, shuffle=True)

        test_loader = DataLoader(
            ImageNet(root=cache_dir, split='val', transform=imagenet_transform),
            batch_size=self.test_batch_size, shuffle=False)

        return {'train_loader': train_loader, 'test_loader': test_loader}


class cifar10(vision_dataloader):

    def __init__(self, name='cifar10', train_batch_size=64, test_batch_size=64):
        super().__init__(name=name)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def load(self, cache_dir='./data/', *args, **kwargs):
        transform = Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            torch.flatten
        ])

        train_loader = DataLoader(
            CIFAR10(root=cache_dir, train=True, download=True, transform=transform),
            batch_size=self.train_batch_size, shuffle=True)

        test_loader = DataLoader(
            CIFAR10(root=cache_dir, train=False, download=True, transform=transform),
            batch_size=self.test_batch_size, shuffle=False)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return {'train_loader': train_loader, 'test_loader': test_loader, 'classes': classes}


class mnist(vision_dataloader):

    def __init__(self, name='mnist', train_batch_size=64, test_batch_size=64):
        super().__init__(name=name)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def load(self, cache_dir='./data/', *args, **kwargs):
        transform = Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            torch.flatten
        ])

        train_loader = DataLoader(
            MNIST(root=cache_dir, train=True, download=True, transform=transform),
            batch_size=self.train_batch_size, shuffle=True)

        test_loader = DataLoader(
            MNIST(root=cache_dir, train=False, download=True, transform=transform),
            batch_size=self.test_batch_size, shuffle=False)

        return {'train_loader': train_loader, 'test_loader': test_loader}
