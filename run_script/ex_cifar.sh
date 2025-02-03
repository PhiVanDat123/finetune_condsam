

# Experiments on Cifar10 with AgnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --batch_size 128 --bz_val 32 --rho 0.05 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --model_name densenet121 --batch_size 128 --bz_val 32 --rho 0.05 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --model_name pyramid101 --batch_size 128 --bz_val 32 --rho 0.05 --training_mode agnosticSAM


# Experiments on Cifar100 with AgnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --batch_size 128 --bz_val 32 --rho 0.1 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name densenet121 --batch_size 128 --bz_val 32 --rho 0.1 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name pyramid101 --batch_size 128 --bz_val 32 --rho 0.1 --training_mode agnosticSAM


# Experiments on Cifar10 with Agnostic-ASAM

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --adaptive --batch_size 128 --bz_val 32 --rho 0.5 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --model_name densenet121 --adaptive --batch_size 128 --bz_val 32 --rho 0.5 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --model_name pyramid101 --adaptive --batch_size 128 --bz_val 32 --rho 0.5 --training_mode agnosticSAM


# Experiments on Cifar100 with Agnostic-ASAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --adaptive --batch_size 128 --bz_val 32 --rho 1.0 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name densenet121 --adaptive --batch_size 128 --bz_val 32 --rho 1.0 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name pyramid101 --adaptive --batch_size 128 --bz_val 32 --rho 1.0 --training_mode agnosticSAM


# Experiments on Cifar10 with SAM

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --batch_size 128 --rho 0.05 --training_mode sam

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --model_name pyramid101 --batch_size 128 --rho 0.05 --training_mode sam

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --model_name densenet121 --batch_size 128 --rho 0.05 --training_mode sam


# Experiments on Cifar100 with SAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --batch_size 128 --rho 0.1 --training_mode sam

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name pyramid101 --batch_size 128 --rho 0.1 --training_mode sam

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name densenet121 --batch_size 128 --rho 0.1 --training_mode sam


# Experiments on Cifar10 with ASAM

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --batch_size 128 --rho 0.5 --training_mode sam

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --model_name pyramid101 --batch_size 128 --rho 0.5 --training_mode sam

python main.py  --dataset_path ../dataset/ --dataset cifar10 --epochs 200 --model_name densenet121 --batch_size 128 --rho 0.5 --training_mode sam


# Experiments on Cifar100 with ASAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --batch_size 128 --rho 1.0 --training_mode sam

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name pyramid101 --batch_size 128 --rho 1.0 --training_mode sam

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name densenet121 --batch_size 128 --rho 1.0 --training_mode sam
