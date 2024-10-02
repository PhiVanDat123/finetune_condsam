
# Ablation study of rho1

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho_lst 0.025_0.1 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho_lst 0.05_0.1 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho_lst 0.1_0.1 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho_lst 0.15_0.1 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.1 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho_lst 0.25_0.1 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho_lst 0.3_0.1 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho_lst 0.35_0.1 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho_lst 0.4_0.1 --training_mode agnosticSAM 


# Ablation study of rho2

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.025 --training_mode agnosticSAM

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.05 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.1 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.15 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.2 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.25 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.3 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.35 --training_mode agnosticSAM 

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 \
--batch_size 512 --bz_val 128 --rho_lst 0.2_0.4 --training_mode agnosticSAM 