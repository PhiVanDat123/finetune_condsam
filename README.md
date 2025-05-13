# Align-SAM: Seeking Flatter Minima for Better Train-Validation Alignment
This is the official implementation of Align-SAM

## Environment
Using Anaconda to install  
`conda env create -f environment.yml`

## Dataset
Create a folder `../dataset` to save downloaded datasets, when you run experiments, datasets will automaticaly download to this folder

## Training model
We provide some training logs in folder [log_files](log_files).
  
The script for all experiments is located in the folder [run_script](run_script).

We list some examples of command lines to train models below:

### Experiments with ImageNet dataset

```bash

# Experiments of ImageNet fine-tuning with Align-SAM
python main.py --dataset_path ../dataset --model_name resnet18 --dataset imagenet --epochs 50 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho 0.05 --training_mode alignSAM --learning_rate 0.01 --weight_decay 0.0001 --pretrain

python main.py --dataset_path ../dataset --model_name resnet34 --dataset imagenet --epochs 50 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho 0.05 --training_mode alignSAM --learning_rate 0.01 --weight_decay 0.0001 --pretrain

python main.py --dataset_path ../dataset --model_name resnet34 --dataset imagenet --epochs 50 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho 0.05 --training_mode alignSAM --learning_rate 0.01 --weight_decay 0.0001 --pretrain


# Experiments of ImageNet training from scratch with Align-SAM
python main.py --dataset_path ../dataset --dataset imagenet --epochs 200 --model_name resnet18 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho 0.05 --training_mode alignSAM --learning_rate 0.01 --weight_decay 0.0001 

python main.py --dataset_path ../dataset --dataset imagenet --epochs 200 --model_name resnet34 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho 0.05 --training_mode alignSAM --learning_rate 0.01 --weight_decay 0.0001 
```

### Experiments with Cifar dataset under noisy label condition

```bash
# Experiments on Cifar100_noise with Align-SAM

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode alignSAM --label_noise_rate 0.2

 
# Experiments on Cifar10_noise with Align-SAM

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode alignSAM --label_noise_rate 0.2

```

### Acknowledgement
This repository is based on [SAM](https://github.com/davda54/sam), [OT-MDR](https://github.com/anh-ntv/OT_MDR)
