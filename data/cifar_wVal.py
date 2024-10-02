import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class Cifar:
    def __init__(self, batch_size, threads, img_size=32, root=None, dataset="cifar10", 
                 pre_trained=False, bz_test=None, custom={},
                 train_transform=None, test_transform=None, total_val=0, 
                 bz_val=30, num_chunk_val=1, correct_split=False):
        """

        :param batch_size:
        :param threads:
        :param root:
        :param dataset:
        :param pre_trained:
        :param bz_test:
        :param custom: {'type': "fft", "data": [ ]}
        """
        if bz_test is None:
            bz_test = batch_size
        if not root:
            root = '../dataset'
        
        if train_transform is None:
            if dataset == "cifar10":
                mean = (0.4914, 0.4822, 0.4465)
                std = (0.2471, 0.2435, 0.2616)
            elif dataset == "cifar100":
                mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
                std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            else:
                mean, std = self._get_statistics(root, dataset=dataset)
            train_transformer_lst = [
                transforms.RandomCrop(size=(img_size, img_size), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
            train_transform = transforms.Compose(train_transformer_lst)

            test_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        if dataset == "cifar10":
            full_train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        elif dataset == "cifar100":
            full_train_set = torchvision.datasets.CIFAR100(root=root,
                                         train=True,
                                         download=True,
                                         transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root=root,
                                        train=False,
                                        download=True,
                                        transform=test_transform)
            self.classes = full_train_set.classes
        total_train_num = len(full_train_set)
        if total_val < 1:
            total_val = int(total_val*total_train_num)
        if total_val == 0:
            total_val = 1

        self.train = torch.utils.data.DataLoader(full_train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.val = torch.utils.data.DataLoader(full_train_set, batch_size=bz_val, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=bz_test, shuffle=False, num_workers=threads)
