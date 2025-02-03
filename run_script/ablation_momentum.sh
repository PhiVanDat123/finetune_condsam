# Ablation study of momentum factor beta

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode sam


python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 1.0 --training_mode sam --adaptive


python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM --agsam_mom 0.9

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM --agsam_mom 0.7


python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM --agsam_mom 0.5


python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM --agsam_mom 0.3

python main.py  --dataset_path ../dataset/ --dataset cifar100 --epochs 200 --model_name resnet32 --random_seed 123 \
--batch_size 512 --bz_val 128 --rho 0.1 --training_mode agnosticSAM --agsam_mom 0.0
