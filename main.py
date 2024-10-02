import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy, Unhinged
from model.pyramidNet import PyramidNet

from model.utils_arch import get_ensemble, EnsembleWrap
from model.resnet_small import resnet32, resnet20, resnet56, resnet110
from data.cifar_wVal import Cifar
from data.aircraft import AirCraft_dataset
from data.cifar_wVal_wNoisyLabel import Cifar_wNoisyLabel
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR

from optimizer.sam import SAM, FriendlySAM
from optimizer.AgnosticSAM import AgnosticSAM

import torchvision.transforms as transforms

def get_dataset(args):
    tran_trans = None
    test_trans = None
    if "imagenet" in args.dataset:
        tran_trans = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.RandomCrop(224, padding=14),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        test_trans = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        
    if args.dataset in ["cifar10", "cifar100"]:
        dataset = Cifar(args.batch_size, args.threads, img_size=args.img_size, root=args.dataset_path, dataset=args.dataset,
                        bz_test=args.bz_test, train_transform=tran_trans, test_transform=test_trans, bz_val=args.bz_val)
    elif args.dataset in ["cifar10_noise", "cifar100_noise"]:
        dataset = Cifar_wNoisyLabel(args.batch_size, args.threads, img_size=args.img_size, root=args.dataset_path, dataset=args.dataset,
                        bz_test=args.bz_test, train_transform=tran_trans, test_transform=test_trans, bz_val=args.bz_val, noise_rate=args.label_noise_rate)
    else:
        args.img_size = 224
        dataset = AirCraft_dataset(args.batch_size, args.threads, img_size=args.img_size, root=args.dataset_path, dataset=args.dataset,
                        bz_test=args.bz_test, train_transform=tran_trans, test_transform=test_trans, bz_val=args.bz_val)
    return dataset


def save_checkpoint(dir, epoch, name="checkpoint", replace=False, **kwargs):
    state = {}
    state.update(kwargs)
    if replace:
        for root, dirs, files in os.walk(dir):
            for f_n in files:
                if name in f_n:
                    os.remove(os.path.join(root, f_n))
                    print("Remove", f_n)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)


def load_checkpoint(dir, model, optimizer, epoch=-1):
    
    filepath = None
    if epoch > -1:
        filepath = os.path.join(dir, "%s-%d.pt" % ("checkpoint", epoch))
    else:
        for root, dirs, files in os.walk(dir):
            for f_n in files:
                if "checkpoint" in f_n:
                    filepath = os.path.join(root, f_n)
                    epoch = int(f_n.split(".pt")[0].split("-")[-1])
    if filepath is not None and os.path.exists(filepath):
        print("Load model from", filepath)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # import pdb; pdb.set_trace()
        # optimizer.state = checkpoint["optimizer_state"]

    return model, optimizer, epoch + 1
    

def lr_schedule(args, optimizer):
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    if args.lr_schedule in ['cosine']:
        scheduler = CosineAnnealingLR(optimizer, args.epochs+1)
    else:
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs+1)
    return scheduler

def train_agnosticSAM(args, model, dataset):

    print("===\nTrain with train_agnosticSAM\n===")
    base_optimizer = torch.optim.SGD
    if args.base_op == "adam":
        base_optimizer = torch.optim.Adam
        optimizer = AgnosticSAM(model.parameters(), base_optimizer, adaptive=args.adaptive,
                              lr=args.learning_rate, weight_decay=args.weight_decay, rho_lst=args.rho_lst, agsam_mom=args.agsam_mom)
    else:
        base_optimizer = torch.optim.SGD
        optimizer = AgnosticSAM(model.parameters(), base_optimizer, adaptive=args.adaptive,
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, rho_lst=args.rho_lst, agsam_mom=args.agsam_mom)
    
    optimizer.assign_name(model)
    str_e = 0
    loss_fn = smooth_crossentropy
    if args.fine_tune:
        model, optimizer, str_e = load_checkpoint(args.log_dir, model, optimizer)
        evaluate(model, loss_fn, dataset, log)
    if args.eval_only: return
    
    log.epoch = str_e - 1
    scheduler = lr_schedule(args, optimizer)
    idx_i_total = 0
    for epoch in range(str_e, args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        
        for idx_i, (batch_val, batch_train) in enumerate(zip(dataset.val, dataset.train)):
            input_val = batch_val[0].to(device)
            target_val = batch_val[1].to(device)
            input_t = batch_train[0].to(device)
            target_t = batch_train[1].to(device)
            
            bz_instance = len(input_t)

            predictions_train = []
            target_train = []
            loss_train = []
            sharpness_train = []
            
            enable_running_stats(model)
            predictions_b1 = model(input_t)
            loss_c1 = loss_fn(predictions_b1, target_t)
            loss_c1.mean().backward()
            optimizer.grad_store(zero_grad=True, grad_name="grad_t", store_model=True)

            predictions_v = model(input_val)
            loss_v = loss_fn(predictions_v, target_val)
            loss_v.mean().backward()
            optimizer.grad_store(zero_grad=True, grad_name="v_update", is_val=True)
            
            optimizer.first_step(zero_grad=True, debug=True if idx_i % 300 == 0 and args.debug else False)

            disable_running_stats(model)
            predictions_b2 = model(input_t)
            loss_max = loss_fn(predictions_b2, target_t)
            loss_max.mean().backward()
            optimizer.second_step(zero_grad=True, debug=True if idx_i % 300 == 0 and args.debug else False)

            predictions_train.append(predictions_b1)
            target_train.append(target_t)
            loss_train.append(loss_c1)
            sharpness = loss_max - loss_c1
            sharpness_train.append(sharpness)
            idx_i_total += 1
            with torch.no_grad():
                predictions = torch.cat(predictions_train)
                targets_merge = torch.cat(target_train)
                correct = torch.argmax(predictions.data, 1) == targets_merge
                if args.lr_schedule in ['cosine']:
                    lr = scheduler.get_lr()[0]
                else:
                    lr = scheduler.lr()
                loss = torch.cat(loss_train)
                log(model, loss.cpu(), correct.cpu(), lr, sharpness=torch.cat(sharpness_train))
        if args.lr_schedule in ['cosine']: scheduler.step()
        else: scheduler(epoch)
        is_eval = True
        if args.dataset == "imagenet" and (epoch + 1) % 5 != 0:
            is_eval = False
        if is_eval:
            model.eval()
            log.flush()
            log.eval(len_dataset=len(dataset.test))
            topk_correct = 0
            num_data = 0
            with torch.no_grad():
                prediction_test = []
                target_test = []
                for batch in dataset.test:
                    inputs, targets = (b.to(device) for b in batch)

                    predictions = model(inputs)
                    loss_c = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    prediction_test.append(nn.functional.softmax(predictions, dim=-1))
                    target_test.append(targets)

                    log(model, loss_c.cpu(), correct.cpu())
                    _, pre_topk = predictions.topk(5, 1, largest=True, sorted=False)
                    topk_correct += 100*(pre_topk == targets.unsqueeze(dim=1)).sum().item()
                    num_data += len(inputs)
            log.flush()
            print("Acc top 5: {:.4f}".format(topk_correct/num_data))

        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            save_checkpoint(args.log_dir, epoch=epoch,
                            name="checkpoint_{}".format(args.random_seed),
                            replace=True,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            optimizer_state = optimizer.state)

   
def train_sam(args, model, dataset):
    print("===\nTrain with train_sam\n===")
    if args.base_op == "adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive,
                    lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        base_optimizer = torch.optim.SGD
        if args.training_mode == "fsam":
            optimizer = FriendlySAM (model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive,
                    lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, lmbda=0.6)
        else:
            optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive,
                    lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.assign_name(model)
    loss_fn = smooth_crossentropy
    str_e = 0
    if args.fine_tune:
        model, optimizer, str_e = load_checkpoint(args.log_dir, model, optimizer)
        evaluate(model, loss_fn, dataset, log)
    
    if args.eval_only:
        return

    log.epoch = str_e - 1
    scheduler = lr_schedule(args, optimizer)
        
    for epoch in range(str_e, args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        iter = 0
        for batch in dataset.train:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            # bz_instance = inputs.size(0)

            enable_running_stats(model)
            predictions = model(inputs)
            loss_c = loss_fn(predictions, targets)
            loss_c.mean().backward()  # update "running_mean" and "running_var"
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            predictions_max = model(inputs)
            loss_max = loss_fn(predictions_max, targets)
            loss_max.mean().backward()  # not update "running_mean" and "running_var"
            optimizer.second_step(zero_grad=True)

            sharpness = loss_max - loss_c
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                if args.lr_schedule in ['cosine']:
                    lr = scheduler.get_lr()[0]
                    if iter == 0:
                        scheduler.step()
                else:
                    lr = scheduler.lr()
                    scheduler(epoch)
                log(model, loss_c.cpu(), correct.cpu(), lr, sharpness=sharpness)
            iter += 1
        is_eval = True
        if args.dataset == "imagenet" and (epoch + 1) % 5 != 0:
            is_eval = False
        if is_eval:
            model.eval()
            log.flush()
            log.eval(len_dataset=len(dataset.test))
            topk_correct = 0
            num_data = 0
            with torch.no_grad():
                for batch in dataset.test:
                    inputs, targets = (b.to(device) for b in batch)

                    predictions = model(inputs)
                    loss_c = loss_fn(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss_c.cpu(), correct.cpu())

                    _, pre_topk = predictions.topk(5, 1, largest=True, sorted=False)
                    topk_correct += 100*(pre_topk == targets.unsqueeze(dim=1)).sum().item()
                    num_data += len(inputs)
            log.flush()
            print("Acc top 5: {:.4f}".format(topk_correct/num_data))
    
        # if (epoch + 1) % 50 == 0 and args.log_dir:
        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            save_checkpoint(args.log_dir, epoch=epoch,
                            name="checkpoint_{}".format(args.random_seed),
                            replace=True,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            optimizer_state = optimizer.state)

def evaluate(model, loss_fn, dataset, log):
    model.eval()
    log.flush()
    log.eval(len_dataset=len(dataset.test))
    topk_correct = 0
    num_data = 0
    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            loss_c = loss_fn(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            log(model, loss_c.cpu(), correct.cpu())

            _, pre_topk = predictions.topk(5, 1, largest=True, sorted=False)
            topk_correct += 100*(pre_topk == targets.unsqueeze(dim=1)).sum().item()
            num_data += len(inputs)
    log.flush()
    print("Acc top 5: {:.4f}".format(topk_correct/num_data))

def train_sgd(args, model, dataset):
    print("===\nTrain with train_sgd\n===")
    
    if args.base_op == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay )
    if args.lr_schedule in ['cosine']:
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    # if args.fine_tune:

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        idx_i = 0
        for batch in dataset.train:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            bz_instance = inputs.size(0)

            predictions = model(inputs)
            loss_c = smooth_crossentropy(predictions, targets)
            loss_c.mean().backward()  # update "running_mean" and "running_var"
            optimizer.step()
            optimizer.zero_grad()

            sharpness = loss_c - loss_c
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                if args.lr_schedule in ['cosine']:
                    lr = scheduler.get_lr()[0]
                    if idx_i == 0:
                        scheduler.step()
                else:
                    lr = scheduler.lr()
                    scheduler(epoch)
                log(model, loss_c.cpu(), correct.cpu(), lr, sharpness=sharpness)
            idx_i += 1

        evaluate(model, smooth_crossentropy, dataset, log)
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        save_checkpoint(args.log_dir, epoch=epoch,
                        name="checkpoint_{}".format(args.random_seed),
                        replace=True,
                        state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(),)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", action="store_true", help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--bz_test", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--bz_val", type=int, default=32, help="Batch size of validation set")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=8, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")

    parser.add_argument("--dataset_path", default="/home/ubuntu/vit_selfOT/ViT-pytorch/data", type=str, help="link to dataset")

    parser.add_argument("--dataset", type=str, default="cifar10", help="name of dataset to load")
    parser.add_argument("--model_name", type=str, default="WRN", help="model_name to load")
    parser.add_argument("--lr_schedule", type=str, default="cosine", help="lr_schedule: step, cosine")
    parser.add_argument("--rho_lst", type=str, default="0.1_0.05", help="using format: rho1_rho2 to set perturbation radius for AgnosticSAM")
    parser.add_argument("--img_size", type=int, default=32, help="Size of input image")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default=None, help="folder to save model")
    parser.add_argument("--fine_tune", action="store_true", help="fine tunning model")
    parser.add_argument("--training_mode", type=str, default="sam", help="training mode: sgd, sam, fsam, agnosticSAM")
    parser.add_argument("--label_noise_rate", type=float, default=0.0)

    parser.add_argument("--pretrain", action="store_true", help="using pretrained model")
    parser.add_argument("--base_op", type=str, default="sgd")
    parser.add_argument("--eval_only", action="store_true", help="Only eval model")
    parser.add_argument("--debug", action="store_true", help="Only eval model")
    parser.add_argument("--agsam_mom", type=float, default=0.9)
    
    args = parser.parse_args()
    args.rho_lst = [float(it) for it in args.rho_lst.split("_")]
    
    for arg_item in vars(args):
        print(arg_item, getattr(args, arg_item))

    initialize(args, seed=args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # import pdb; pdb.set_trace()
    dataset = get_dataset(args)
    log = Log(log_each=10)
    num_cls = len(dataset.classes)
    if args.model_name == "WRN":
        model = WideResNet(28, 10, 0.0, in_channels=3, labels=num_cls)
    elif args.model_name == "densenet121":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False, num_classes=num_cls)
    elif args.model_name == "resnet18":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=args.pretrain, num_classes=num_cls)
    elif args.model_name == "resnet34":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=args.pretrain, num_classes=num_cls)
    elif args.model_name == "resnet50":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=args.pretrain, num_classes=num_cls)
    elif args.model_name == "resnet32":
        model = resnet32(num_classes=num_cls)
    elif args.model_name == "pyramid101":
        model = PyramidNet(dataset=args.dataset, depth=101, alpha=64, num_classes=num_cls, bottleneck=True)
    elif args.model_name == "EfficientNet-B4":
        if args.pretrain:
            model = torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_cls),
            )
        else:
            model = torchvision.models.efficientnet_b4(num_classes=num_cls)
        
    elif args.model_name == "EfficientNet-B3":
        if args.pretrain:
            model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_cls),
            )
        else:
            model = torchvision.models.efficientnet_b3(num_classes=num_cls)
        
    elif args.model_name == "EfficientNet-B2":
        # model = torchvision.models.efficientnet_b2(num_classes=num_cls)
        if args.pretrain:
            model = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_cls),
            )
        else:
            model = torchvision.models.efficientnet_b2(num_classes=num_cls)

    model.to(device)

    if args.training_mode in ["agnosticSAM"]:
        train_agnosticSAM(args, model, dataset)
    elif args.training_mode in ["sam", "fsam"]:
        train_sam(args, model, dataset) 
    elif args.training_mode in ["sgd"]:
        train_sgd(args, model, dataset)
    else: print("Please check the training_mode")
    log.flush()
