
# Ablation study of rho1

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM --agsam_lambda 0.25

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM  --agsam_lambda 0.5

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM  --agsam_lambda 1.0

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM  --agsam_lambda 1.5

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM  --agsam_lambda 2.0

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM  --agsam_lambda 2.5

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM  --agsam_lambda 3.0

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM  --agsam_lambda 3.5

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM  --agsam_lambda 4.0

