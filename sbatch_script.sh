#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable          # Partition name
#SBATCH --output=multi_gpu_%j.out     # Standard output and error log
#SBATCH --error=multi_gpu_%j.err      # Error log



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

# Execute Python script

wandb agent korra141/Diff-HEF/oh7y26ru
python mlp_measurement_model_with_analytical_hef_pm_on_s1.py
