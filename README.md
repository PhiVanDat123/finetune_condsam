# Agnostic Sharpness-Aware Minimization
This is the official implementation of AgnosticSAM

## Environment
USing Annaconda to install  
`conda env create -f environment.yml`

## Dataset
Create a folder `../dataset` to save downloaded dataset, when you run experiment, datasets will automaticaly download to this folder

## Training model
We provide some training logs in folder `log_files`.
  
The script for all experiments is located in folder `run_script` except experiments with MAML setting.

### Experiments with CIFAR dataset

```bash
bash run_script/ex_cifar.sh
```

## Experiments with MAML setting

Please check folder `MAML` for experiments on MAML setting.


### Acknowledgement
This repository is based on [SAM](https://github.com/davda54/sam), [OT-MDR](https://github.com/anh-ntv/OT_MDR) and [Sharp-MAML](https://github.com/mominabbass/Sharp-MAML)
