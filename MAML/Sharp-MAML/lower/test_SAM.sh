c_file="/home/jnguyen/Sharp-MAML/lower/logs/SGD/miniimagenet/2024-09-05_135131_omni_sharp_lower_0.05_0.1_5_shots_1_epochs_200/config.json"

for v in 1 2 3 4 5
do
    python test.py --verbose $c_file
done

c_file_se="/home/jnguyen/Sharp-MAML/lower/logs/SGD/miniimagenet/2024-09-05_151659_omni_sharp_lower_0.05_0.1_5_shots_4_epochs_200/config.json"

for v in 1 2 3 4 5
do
    python test.py --verbose $c_file_se
done