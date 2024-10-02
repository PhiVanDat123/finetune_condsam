#!/bin/sh
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=50000
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --nodelist=node04

# Memory usage (MB)

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=2-00:00:00
# To receive an email when job completes or fails
#SBATCH --mail-user=van-anh.nguyen@monash.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/jnguyen/Sharp-MAML/lower_AgSAM/logs/test.out
#SBATCH --error=/home/jnguyen/Sharp-MAML/lower_AgSAM/logs/test.err

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shared/miniconda3/lib/
eval "$(conda shell.bash hook)"
conda activate py39_torchmeta



python test.py /home/jnguyen/Sharp-MAML/lower_AgSAM/logs/AgSAM/miniimagenet/2024-08-29_175855_omni_sharp_lower_0.0005_0.1_5_shots_4_epochs_600/config.json