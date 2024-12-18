#!/bin/bash
#SBATCH --job-name=S1_UE_MM_HED
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --partition=long-cpu   # Partition name
#SBATCH --output=slurm_logs/multi_gpu_%j.out     # Standard output and error log
#SBATCH --error=slurm_logs/multi_gpu_%j.err      # Error log



# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/12.6.0/cudnn/9.3

# Activate pre-existing environment.
source  activate disc_tracking
cd $HOME/scratch/Diff-HEF/scripts

wandb login 8ffe865c4b82a4e1f84ebcf8cc9681892e828854
unset CUDA_VISIBLE_DEVICES


wandb agent korra141/Diff-HEF/3aug2ygm
# Execute Python script
#wandb agent korra141/Diff-HEF/wwntns3h
#ndb agent korra141/Diff-HEF/j97cb60ai
#python cnn_uncertainity_estimation_in_r2.py
