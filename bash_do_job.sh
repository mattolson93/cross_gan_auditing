#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --partition=pvis
#SBATCH -J stylgan2

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda


srun --mpibind=off {\
CUDA_VISIBLE_DEVICES=0 python 2train.py k=16 overlap_k=12 generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=to_rgbs.5 hparams.batch_size=2   generator.class_name=celeba generator2.class_name=metfacesmall model=linear model2=linear  model.alpha=3 model2.alpha=3  generator2.use_w=True generator.use_w=True projector=resnet batch_k=5 projector.layers=5  train_projector=False loss.otherweight=0.1 loss=classification2  projector.name=resnet50 info=SOMETHING & ;

; sleep 1800; exit



