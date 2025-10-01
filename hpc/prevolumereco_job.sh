#!/bin/sh
#
#SBATCH --job-name=prevolumereco
#SBATCH --partition=bare-metal-nodes
#SBATCH --gres=gpu:1
#SBATCH --output=/storage-hpc/ntosi/SAND-LAr-PROD/GRAIN/numu_CC_QE_Filippo/prevolumereco/logs/prevolumereco.out
#SBATCH --error=/storage-hpc/ntosi/SAND-LAr-PROD/GRAIN/numu_CC_QE_Filippo/prevolumereco/logs/prevolumereco.err

source /storage-hpc/ntosi/SAND-LAr-BIN/pre-volumereco-venv/bin/activate
srun /home/HPC/fmeihpc/scripts/pre-volumereco/hpc/train_hpc.py
deactivate
