c_file="/home/jnguyen/Sharp-MAML/lower_AgSAM/logs/AgSAM/miniimagenet/2024-09-05_013159_omni_sharp_lower_0.005_0.1_5_shots_1_epochs_200/config.json"

for v in 1 2 3 4 5
do
    python test.py --verbose $c_file
done

c_file_se="/home/jnguyen/Sharp-MAML/lower_AgSAM/logs/AgSAM/miniimagenet/2024-09-05_013940_omni_sharp_lower_0.005_0.1_5_shots_4_epochs_200/config.json"

for v in 1 2 3 4 5
do
    python test.py --verbose $c_file_se
done