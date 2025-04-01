**Table 1: Classification accuracy on the CIFAR datasets with the same initialized weights on ResNet32**
| Method | Cifar-10 | Cifar-100 |
|---|---|---|
| SAM | 93.01 | 70.27 |
| AgSAM | **93.34** | **70.67** |

&nbsp;

**Table 2: Domain Generalization setting. All models are trained on ImageNet-1k and then evaluated on ImageNet-1k (clean validation set), Imagenet-R, and Imagenet-C datasets.**
|  |  | Top-1 Acc |  |  | Top-5 Acc |  |
|---|---|---|---|---|---|---|
|  | Imagenet | ImageNet-R | ImageNet-C | Imagenet | ImageNet-R | ImageNet-C |
| **ResNet18 - Transfer Learning** |  |  |  |  |  |  |
| SAM | 70.52 | 34.18 | 48.27 | 89.60 | 52.82 |  72.17 |
| AgSAM | **70.88** | **34.38** | **48.69** | **89.94** | **53.23** | **72.61** |
| **ResNet18 - Train From Scratch** |  |  |  |  |  |  |
| SAM | 62.46 | 25.86 | 32.96 | 73.15 | 43.09 | 55.72 |
| AgSAM | **63.64** | **26.20** | **34.06** | **73.45** | **43.99** | **57.42** |

&nbsp;

**Table 3: Additional experiments on CIFAR datasets with GSAM and LookSAM.**
| Method | WideResnet28x10 | Pyramid101 | Densenet121 |
|---|---|---|---|
| **Dataset CIFAR-100** |  |  |
| GSAM | 83.13 |  81.87 |  68.88 |
| LookSAM (k=1) | 82.89 | 82.25  |  69.05  |
| AgSAM | **83.72** | **82.53** | **69.10** |
| **Dataset CIFAR-10** |  |  |
| GSAM | 96.91  |  96.15  | 91.50 |
| LookSAM (k=1) | **97.00**   |   96.00  |   91.10 |
| AgSAM | 96.91  | **96.47**  | **91.54** |
