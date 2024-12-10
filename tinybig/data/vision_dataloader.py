# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
# Vision Dataloader #
#####################

from abc import abstractmethod

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import transforms, Compose, ToTensor, Normalize, Lambda
from torchvision.datasets import MNIST, CIFAR10, ImageNet

from tinybig.data.base_data import dataloader


class vision_dataloader(dataloader):
    """
    Base class for vision-based dataloaders.

    This class serves as a template for vision-based dataloaders, providing an interface for loading datasets.

    Attributes
    ----------
    name : str
        Name of the dataloader.
    train_batch_size : int
        Batch size for training data.
    test_batch_size : int
        Batch size for testing data.

    Methods
    -------
    __init__(...)
        Initializes the vision dataloader.
    load()
        Abstract method for loading the dataset; must be implemented in subclasses.
    """
    def __init__(self, train_batch_size: int, test_batch_size: int,  name: str = 'vision_dataloader', *args, **kwargs):
        """
        Initializes the vision dataloader.

        Parameters
        ----------
        train_batch_size : int
            Batch size for training data.
        test_batch_size : int
            Batch size for testing data.
        name : str, optional
            Name of the dataloader, default is 'vision_dataloader'.
        *args, **kwargs
            Additional arguments for the parent class initialization.

        Returns
        -------
        None
        """
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    @abstractmethod
    def load(self):
        """
        Abstract method for loading the dataset.

        This method must be implemented in subclasses to handle dataset-specific loading logic.

        Returns
        -------
        dict
            A dictionary containing the train and test dataloaders.
        """
        pass


class imagenet(vision_dataloader):
    """
    A dataloader for the ImageNet dataset.

    Handles loading and preprocessing of the ImageNet dataset.

    Attributes
    ----------
    name : str
        Name of the dataloader, default is 'imagenet'.
    train_batch_size : int
        Batch size for training data.
    test_batch_size : int
        Batch size for testing data.

    Methods
    -------
    __init__(...)
        Initializes the ImageNet dataloader.
    load(...)
        Loads and preprocesses the ImageNet dataset.
    """
    def __init__(self, name='imagenet', train_batch_size: int = 64, test_batch_size: int = 64):
        """
        Initializes the ImageNet dataloader.

        Parameters
        ----------
        name : str, optional
            Name of the dataloader, default is 'imagenet'.
        train_batch_size : int, optional
            Batch size for training data, default is 64.
        test_batch_size : int, optional
            Batch size for testing data, default is 64.

        Returns
        -------
        None
        """
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    # @staticmethod
    # def flatten(x):
    #     x = torch.flatten(x)
    #     return x.view(-1)

    def load(self, cache_dir='./data/', with_transformation: bool = True, *args, **kwargs):
        """
        Loads and preprocesses the ImageNet dataset.

        Parameters
        ----------
        cache_dir : str, optional
            Directory to cache the dataset, default is './data/'.
        with_transformation : bool, optional
            Whether to load the training or testing dataset with transformation, default is True.
        *args, **kwargs
            Additional arguments for dataset loading.

        Returns
        -------
        dict
            A dictionary containing the train and test dataloaders.
        """
        if with_transformation:
            imagenet_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                torch.flatten
            ])
        else:
            imagenet_transform = Compose([
                transforms.ToTensor(),
            ])

        train_loader = DataLoader(
            ImageNet(root=cache_dir, split='train', transform=imagenet_transform),
            batch_size=self.train_batch_size, shuffle=True)

        test_loader = DataLoader(
            ImageNet(root=cache_dir, split='val', transform=imagenet_transform),
            batch_size=self.test_batch_size, shuffle=False)

        return {'train_loader': train_loader, 'test_loader': test_loader}


class cifar10(vision_dataloader):
    """
    A dataloader for the CIFAR-10 dataset.

    Handles loading and preprocessing of the CIFAR-10 dataset.

    Attributes
    ----------
    name : str
        Name of the dataloader, default is 'cifar10'.
    train_batch_size : int
        Batch size for training data.
    test_batch_size : int
        Batch size for testing data.

    Methods
    -------
    __init__(...)
        Initializes the CIFAR-10 dataloader.
    load(...)
        Loads and preprocesses the CIFAR-10 dataset.
    """
    def __init__(self, name='cifar10', train_batch_size: int = 64, test_batch_size: int = 64):
        """
        Initializes the CIFAR-10 dataloader.

        Parameters
        ----------
        name : str, optional
            Name of the dataloader, default is 'cifar10'.
        train_batch_size : int, optional
            Batch size for training data, default is 64.
        test_batch_size : int, optional
            Batch size for testing data, default is 64.

        Returns
        -------
        None
        """
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    def load(self, cache_dir='./data/', with_transformation: bool = True, *args, **kwargs):
        """
        Loads and preprocesses the CIFAR-10 dataset.

        Parameters
        ----------
        cache_dir : str, optional
            Directory to cache the dataset, default is './data/'.
        with_transformation : bool, optional
            Whether to load the training or testing dataset with transformation, default is True.
        *args, **kwargs
            Additional arguments for dataset loading.

        Returns
        -------
        dict
            A dictionary containing the train and test dataloaders, and the class labels.
        """
        if with_transformation:
            transform = Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                torch.flatten
            ])
        else:
            transform = Compose([
                transforms.ToTensor(),
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
    """
    A dataloader for the MNIST dataset.

    Handles loading and preprocessing of the MNIST dataset.

    Attributes
    ----------
    name : str
        Name of the dataloader, default is 'mnist'.
    train_batch_size : int
        Batch size for training data.
    test_batch_size : int
        Batch size for testing data.

    Methods
    -------
    __init__(...)
        Initializes the MNIST dataloader.
    load(...)
        Loads and preprocesses the MNIST dataset.
    """
    def __init__(self, name='mnist', train_batch_size: int = 64, test_batch_size: int = 64):
        """
        Initializes the MNIST dataloader.

        Parameters
        ----------
        name : str, optional
            Name of the dataloader, default is 'mnist'.
        train_batch_size : int, optional
            Batch size for training data, default is 64.
        test_batch_size : int, optional
            Batch size for testing data, default is 64.

        Returns
        -------
        None
        """
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    def load(self, cache_dir='./data/', with_transformation: bool = True, *args, **kwargs):
        """
        Loads and preprocesses the MNIST dataset.

        Parameters
        ----------
        cache_dir : str, optional
            Directory to cache the dataset, default is './data/'.
        with_transformation : bool, optional
            Whether to load the training or testing dataset with transformation, default is True.
        *args, **kwargs
            Additional arguments for dataset loading.

        Returns
        -------
        dict
            A dictionary containing the train and test dataloaders.
        """

        if with_transformation:
            transform = Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Normalize((0.1307,), (0.3081,)),
                torch.flatten
            ])
        else:
            transform = Compose([
                transforms.ToTensor(),
            ])

        train_loader = DataLoader(
            MNIST(root=cache_dir, train=True, download=True, transform=transform),
            batch_size=self.train_batch_size, shuffle=True)

        test_loader = DataLoader(
            MNIST(root=cache_dir, train=False, download=True, transform=transform),
            batch_size=self.test_batch_size, shuffle=False)

        return {'train_loader': train_loader, 'test_loader': test_loader}
