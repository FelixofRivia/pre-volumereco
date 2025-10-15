#!/bin/sh
#
#SBATCH --job-name=prevolumereco
#SBATCH --partition=bare-metal-nodes
#SBATCH --output=/storage-hpc/ntosi/SAND-LAr-PROD/GRAIN/numu_CC_QE_Filippo/prevolumereco/logs/prevolumereco_CPU_10cm.out
#SBATCH --error=/storage-hpc/ntosi/SAND-LAr-PROD/GRAIN/numu_CC_QE_Filippo/prevolumereco/logs/prevolumereco_CPU_10cm.err

source /storage-hpc/ntosi/SAND-LAr-BIN/pre-volumereco-venv/bin/activate
time python3 /home/HPC/fmeihpc/scripts/pre-volumereco/hpc/train_hpc.py /storage-hpc/ntosi/SAND-LAr-PROD/GRAIN/numu_CC_QE_Filippo/training_dataset/dataset_10cm.h5 --output_dir /storage-hpc/ntosi/SAND-LAr-PROD/GRAIN/numu_CC_QE_Filippo/trained_models --model_name model_10cm --n_trials 300 --cpu_only
deactivate
