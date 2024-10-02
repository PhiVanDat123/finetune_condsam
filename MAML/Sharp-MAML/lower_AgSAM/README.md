# Agnostic-SAM for lower-loss in meta learning

An implementation of Agnostic-SAM to minize the lower-level loss during training meta-model.


### Preparation
Download the Onmiglot dataset from [this link](https://drive.google.com/drive/folders/1OT8mNSKoTvhgT3dE1g545LrPuUjLVcfJ?usp=sharing). Unzip the datasets and place them into a folder.


Follow instructions from [Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) to download the Mini-imagenet dataset.  Unzip the datasets and place them into a folder.

### Usage
You can use [`train_mini-sharplower.py`](train_mini-sharplower.py) to meta-train your model with Agnostic-SAM on lower-level loss.

- Experiments on **Miniimagenet 5-way 1-shot**, run:
```bash
python train_mini-sharplower.py ../dataset --dataset miniimagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.1 --batch-size 4 --num-workers 8 --num-epochs 600 --output-folder logs/miniimagenet_5way1shot --num-steps 5 --alpha 0.005
```

- Experiments on **Miniimagenet 5-way 5-shot**, run:
```bash
python train_mini-sharplower.py ../dataset --dataset miniimagenet --num-ways 5 --num-shots 5 --use-cuda --step-size 0.1 --batch-size 4 --num-workers 8 --num-epochs 600 --output-folder logs/miniimagenet_5way5shot --num-steps 5 --alpha 0.005
```

- Experiments on **Omniglot 20-way 1-shot**, run:
```bash
python train_mini-sharplower.py ../dataset --dataset omniglot --num-ways 20 --num-shots 1 --use-cuda --step-size 0.1 --batch-size 16 --num-workers 8 --num-epochs 600 --output-folder logs/omniglot_20way1shot --num-steps 3 --alpha 0.0005
```

- Experiments on **Omniglot 20-way 1-shot**, run:
```bash
python train_mini-sharplower.py ../dataset --dataset omniglot --num-ways 20 --num-shots 5 --use-cuda --step-size 0.1 --batch-size 16 --num-workers 8 --num-epochs 600 --output-folder logs/omniglot_20way5shot --num-steps 3 --alpha 0.0005
```

### Evaluation
The meta-training script creates a configuration file you can use to meta-test your model. You can use [`test.py`](test.py) to meta-test your model:
```bash
python test.py logs/omniglot_20way5shot/config.json
```

