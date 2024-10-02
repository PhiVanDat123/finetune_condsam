import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
# from data.cifar_noise import CIFAR10,CIFAR100
from data.cifar_noise_fSAM import get_datasets_cutout

cifar10_mean = (0.4914, 0.4823, 0.4466)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
svhn_mean = (129.3/255, 124.1/255, 112.4/255)
svhn_std = (68.2/255, 65.4/255.0, 70.4/255.0)
imagenet_mean = (0.485,0.456,0.406)
imagenet_std = (0.229,0.224,0.225)

# Credit: https://github.com/LJY-HY/MentorMix_pytorch/blob/master/dataset/cifar.py
def get_transform(dataset='cifar10', mode='train'):
    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean = cifar10_mean, std = cifar10_std)
    elif dataset == 'cifar100':
        normalize = transforms.Normalize(mean = cifar100_mean, std = cifar100_std)
    elif dataset == 'svhn':
        normalize = transforms.Normalize(mean = svhn_mean, std = svhn_std)

    if mode == 'train':
        TF = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding = int(32*0.125)),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == 'test':
        TF = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])
    
    return TF

def cifar10(root, train_TF = None, test_TF = None, noise_rate=0.2):
    if train_TF is None and test_TF is None:
        train_TF = get_transform("cifar10", 'train')
        test_TF = get_transform("cifar10", 'test')
  
    train_dataset = CIFAR10(root = root, train_MentorNet = False, 
                            corruption_prob=noise_rate, train=True, 
                            corruption_type='flip',
                            # corruption_type='unif',
                            transform = train_TF, download=True)
    test_dataset = CIFAR10(root = root, train=False, transform = test_TF, download=False)
    return train_dataset, test_dataset

def cifar100(root, train_TF = None, test_TF = None, noise_rate=0.2):
    if train_TF is None and test_TF is None:
        train_TF = get_transform("cifar10", 'train')
        test_TF = get_transform("cifar10", 'test')
  
    train_dataset = CIFAR100(root = root, train_MentorNet = False, 
                            corruption_prob=noise_rate, train=True, 
                            corruption_type='flip',
                            # corruption_type='unif',
                            transform = train_TF, download=True)
    test_dataset = CIFAR100(root = root, train=False, transform = test_TF, download=False)
    return train_dataset, test_dataset


class Cifar_wNoisyLabel:
    def __init__(self, batch_size, threads, img_size=32, root=None, dataset="cifar10", cutout=False, pre_trained=False, bz_test=None, custom={},
                 train_transform=None, test_transform=None, total_val=500, bz_val=30, num_chunk_val=1, noise_rate=0.2):
        # train_TF = None, test_TF = None, noise_rate=0.2, batch_size=32
        if dataset == "cifar10_noise":
            # train_dataset, test_dataset = cifar10(root, train_TF=train_transform, test_TF=test_transform, 
            #                                   noise_rate=noise_rate)
            train_dataset, test_dataset = get_datasets_cutout(root, dataset, noise_rate=noise_rate)
            num_classes = 10
        elif dataset == "cifar100_noise":
            # train_dataset, test_dataset = cifar100(root, train_TF=train_transform, test_TF=test_transform, 
            #                                   noise_rate=noise_rate)
            train_dataset, test_dataset = get_datasets_cutout(root, dataset, noise_rate=noise_rate)
            num_classes = 100
        self.train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
        self.val = DataLoader(train_dataset, batch_size = bz_val, shuffle = True, num_workers = 8)
        if bz_test is None:
            bz_test = batch_size
        self.test = DataLoader(test_dataset, batch_size = bz_test, shuffle = False, num_workers = 8)
        self.classes = torch.arange(num_classes)
        self.val_chunk = []

if __name__ == '__main__':
    train_dt, test_dt = cifar10("../../dataset/")
    import pdb; pdb.set_trace()
