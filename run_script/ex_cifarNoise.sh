# Experiments on Cifar100_noise with AgnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode agnosticSAM --label_noise_rate 0.2

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode agnosticSAM --label_noise_rate 0.4

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode agnosticSAM --label_noise_rate 0.6

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode agnosticSAM --label_noise_rate 0.8

 
# Experiments on Cifar10_noise with AgnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode agnosticSAM --label_noise_rate 0.2

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode agnosticSAM --label_noise_rate 0.4

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode agnosticSAM --label_noise_rate 0.6

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 --random_seed 765 \
--batch_size 512 --bz_val 128  --rho 0.1 --training_mode agnosticSAM --label_noise_rate 0.8


# Experiments on Cifar100_noise with SAM
python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode sam --label_noise_rate 0.2 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode sam --label_noise_rate 0.4 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode sam --label_noise_rate 0.6 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.05 --training_mode sam --label_noise_rate 0.8 --random_seed 765


# Experiments on Cifar10_noise with SAM
python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode sam --label_noise_rate 0.2 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode sam --label_noise_rate 0.4 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode sam --label_noise_rate 0.6 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.05 --training_mode sam --label_noise_rate 0.8 --random_seed 765


# Experiments on Cifar100_noise with FSAM
python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode fsam --label_noise_rate 0.2 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode fsam --label_noise_rate 0.4 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode fsam --label_noise_rate 0.6 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar100_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.05 --training_mode fsam --label_noise_rate 0.8 --random_seed 765


# Experiments on Cifar10_noise with FSAM
python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode fsam --label_noise_rate 0.2 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode fsam --label_noise_rate 0.4 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.1 --training_mode fsam --label_noise_rate 0.6 --random_seed 765

python main.py  --dataset_path ../dataset/ --dataset cifar10_noise --epochs 200 --model_name resnet32 \
--batch_size 512  --rho 0.05 --training_mode fsam --label_noise_rate 0.8 --random_seed 765


