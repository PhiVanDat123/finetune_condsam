import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models_imagenet
# from models.pyramidnet import PyramidNet
# from timm.models.vision_transformer import VisionTransformer, _cfg
# from functools import partial
from torch.utils.data import DataLoader, Dataset, Subset

import numpy as np
import random
import os
import time
import sys
# import torch.utils.data as data
# from torchvision.datasets.utils import download_url, check_integrity
import os.path
import pickle
from PIL import Image

import torch.nn.functional as F
# from torch.optim.optimizer import Optimizer, required
import math
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image


class cifar_dataset(Dataset): 
    def __init__(self, dataset='cifar10', r=0.4, noise_mode='sym', root_dir='./datasets/cifar-10-batches-py', 
                 transform=None, mode='all', noise_file='cifar10.json', pred=[], probability=[], log=''): 
        
        self.r = r # noise ratio
        self.transform = transform
        if dataset=='cifar100':
            root_dir = os.path.join(root_dir, 'cifar-100-python')
        elif dataset=='cifar10':
            root_dir = os.path.join(root_dir, 'cifar-10-batches-py')
        self.mode = mode #mode 'test', 'all', 'labeled', 'unlabeled'
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.noise_file = os.path.join(root_dir, noise_file)
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))   #(1000,32,32,3)
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  #(1000,32,32,3)
                self.test_label = test_dic['fine_labels']                            
        else:   #'train
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6): #1~5
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1)) #(5000,32,32,3)

            if os.path.exists(self.noise_file):
                noise_label = json.load(open(self.noise_file,"r"))
            else:    #inject noise   
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            if dataset=='cifar10': 
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)                    
                    else:    
                        noise_label.append(train_label[i])   
                # print("save noisy labels to %s ..."%self.noise_file)   
                # print('self.nose_file', type(self.noise_file), self.noise_file)     
                # json.dump(noise_label,open(self.noise_file,"w"))  
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return (img, target)    
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return (img, target)
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)        

class cifar_dataloader():  
    def __init__(self, dataset='cifar10', r=0.2, noise_mode='sym', batch_size=256, num_workers=4, cutout=False,root_dir='', log='', noise_file='cifar10.json'):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cutout = cutout
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            if self.cutout:
                self.transform_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                        Cutout()
                    ]) 
            else:
                self.transform_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                    ])   
        elif self.dataset=='cifar100':
            if self.cutout:
                self.transform_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                        Cutout()
                        # transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                    ]) 
            else:
                self.transform_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                        # transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                    ]) 
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                # transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])   
    
    def get_loader(self):

        train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, 
                                    root_dir='./datasets/cifar-10-batches-py', transform=self.transform_train, mode="all", 
                                    noise_file=self.noise_file)              
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers, pin_memory=True)   
        
        val_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, 
                                    root_dir='./datasets/cifar-10-batches-py', transform=self.transform_test, mode="test", 
                                    noise_file=self.noise_file)                    
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True)     
        return train_loader, val_loader
    
    def get_dataset(self):

        train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, 
                                    root_dir=self.root_dir, transform=self.transform_train, mode="all", 
                                    noise_file=self.noise_file)              
        # train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
        #                             num_workers=self.num_workers, pin_memory=True)   
        
        val_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, 
                                    root_dir=self.root_dir, transform=self.transform_test, mode="test", 
                                    noise_file=self.noise_file)                    
        # val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True,
        #                         num_workers=self.num_workers, pin_memory=True)     
        return train_dataset, val_dataset
    
def get_datasets_cutout(root, dataset, noise_rate=0.2):
    print ('cutout!')
    if dataset == 'CIFAR10':
        print ('cifar10 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                Cutout()
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    elif dataset == 'cifar10_noise':
        print('cifar10 nosie dataset!')
        cifar10_noise = cifar_dataloader(root_dir=root, dataset='cifar10', r=noise_rate, cutout=True)
        train_set, test_set = cifar10_noise.get_dataset()
 
    elif dataset == 'cifar100_noise':
        print('cifar100 nosie dataset!')
        cifar100_noise = cifar_dataloader(root_dir=root, dataset='cifar100', r=noise_rate, cutout=True)
        train_set, test_set = cifar100_noise.get_dataset()

    elif dataset == 'CIFAR100':
        print ('cifar100 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=root, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                Cutout()
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    return train_set, test_set