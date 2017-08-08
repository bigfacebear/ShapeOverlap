#!/bin/bash
#
#SBATCH --job-name=siamese_trans
#SBATCH --output=/home/xsede/users/xs-qczhao/outputs/trans_siamese.txt
#
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12800
#SBATCH --gres gpu:1
#SBATCH --gres-flags=enforce-binding

module load tensorflow
python /home/xsede/users/xs-qczhao/ShapeOverlap/train_transformer_siamese.py
