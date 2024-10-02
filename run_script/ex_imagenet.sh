# Experiments of ImageNet fine-tuning with AgnosticSAM
python main.py --dataset_path ../dataset --model_name resnet18 --dataset imagenet --epochs 50 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho_lst 0.1_0.05 --training_mode agnosticSAM --learning_rate 0.01 --weight_decay 0.0001 --pretrain

python main.py --dataset_path ../dataset --model_name resnet34 --dataset imagenet --epochs 50 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho_lst 0.1_0.05 --training_mode agnosticSAM --learning_rate 0.01 --weight_decay 0.0001 --pretrain

python main.py --dataset_path ../dataset --model_name resnet34 --dataset imagenet --epochs 50 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho_lst 0.1_0.05 --training_mode agnosticSAM --learning_rate 0.01 --weight_decay 0.0001 --pretrain


# Experiments of ImageNet training from the scratch with AgnosticSAM
python main.py --dataset_path ../dataset --dataset imagenet --epochs 200 --model_name resnet18 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho_lst 0.1_0.05 --training_mode agnosticSAM --learning_rate 0.01 --weight_decay 0.0001 

python main.py --dataset_path ../dataset --dataset imagenet --epochs 200 --model_name resnet34 --batch_size 2048 --bz_val 512 --bz_test 2048  --rho_lst 0.1_0.05 --training_mode agnosticSAM --learning_rate 0.01 --weight_decay 0.0001 


# Experiments of ImageNet fine-tuning with SAM
python main.py --dataset_path ../dataset --dataset imagenet --epochs 50 --model_name resnet18 --batch_size 2048 --bz_test 2048  --rho 0.05 --training_mode sam --learning_rate 0.01 --weight_decay 0.0001 --pretrain

python main.py --dataset_path ../dataset --dataset imagenet --epochs 50 --model_name resnet34 --batch_size 2048 --bz_test 2048  --rho 0.05 --training_mode sam --learning_rate 0.01 --weight_decay 0.0001 --pretrain

python main.py --dataset_path ../dataset --dataset imagenet --epochs 50 --model_name resnet50 --batch_size 2048 --bz_test 2048  --rho 0.05 --training_mode sam --learning_rate 0.01 --weight_decay 0.0001 --pretrain


# Experiments of ImageNet training from the scratch with SAM
python main.py --dataset_path ../dataset --dataset imagenet --epochs 200 --model_name resnet18 --batch_size 2048 --bz_test 2048  --rho 0.05 --training_mode sam --learning_rate 0.01 --weight_decay 0.0001 

python main.py --dataset_path ../dataset --dataset imagenet --epochs 200 --model_name resnet34 --batch_size 2048 --bz_test 2048  --rho 0.05 --training_mode sam --learning_rate 0.01 --weight_decay 0.0001 

