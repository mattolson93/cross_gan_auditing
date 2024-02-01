#!/bin/bash

GAN1=$1
GAN2=$2
GAN3=$3
GPU=0

echo "{ CUDA_VISIBLE_DEVICES=$GPU  python 2train.py k=16 overlap_k=12 generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=2   generator.class_name=$GAN1 generator2.class_name=$GAN2 model=linear model2=linear  model.alpha=3 model2.alpha=3  generator2.use_w=True generator.use_w=True projector=resnet batch_k=5 projector.layers=5  train_projector=False loss.otherweight=0.1  projector.name=resnet50  projector.load_path=advbn_augmix_res50.pth.tar dre_lamb=0.2 ; CUDA_VISIBLE_DEVICES=$GPU  python 2train.py k=16 overlap_k=12 generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=2   generator.class_name=$GAN1 generator2.class_name=$GAN2 model=linear model2=linear  model.alpha=3 model2.alpha=3  generator2.use_w=True generator.use_w=True projector=resnet batch_k=5 projector.layers=5  train_projector=False loss.otherweight=0.1  projector.name=resnet50  projector.load_path=advbn_augmix_res50.pth.tar dre_lamb=0.5 ; CUDA_VISIBLE_DEVICES=$GPU  python 2train.py k=16 overlap_k=12 generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=2   generator.class_name=$GAN1 generator2.class_name=$GAN2 model=linear model2=linear  model.alpha=3 model2.alpha=3  generator2.use_w=True generator.use_w=True projector=resnet batch_k=5 projector.layers=5  train_projector=False loss.otherweight=0.1  projector.name=resnet50  projector.load_path=advbn_augmix_res50.pth.tar dre_lamb=5.0 ;  } & "
GPU=1
echo "{ CUDA_VISIBLE_DEVICES=$GPU  python 2train.py k=16 overlap_k=12 generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=2   generator.class_name=$GAN1 generator2.class_name=$GAN3 model=linear model2=linear  model.alpha=3 model2.alpha=3  generator2.use_w=True generator.use_w=True projector=resnet batch_k=5 projector.layers=5  train_projector=False loss.otherweight=0.1  projector.name=resnet50  projector.load_path=advbn_augmix_res50.pth.tar dre_lamb=0.2 ; CUDA_VISIBLE_DEVICES=$GPU  python 2train.py k=16 overlap_k=12 generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=2   generator.class_name=$GAN1 generator2.class_name=$GAN3 model=linear model2=linear  model.alpha=3 model2.alpha=3  generator2.use_w=True generator.use_w=True projector=resnet batch_k=5 projector.layers=5  train_projector=False loss.otherweight=0.1  projector.name=resnet50  projector.load_path=advbn_augmix_res50.pth.tar dre_lamb=0.5 ; CUDA_VISIBLE_DEVICES=$GPU  python 2train.py k=16 overlap_k=12 generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=2   generator.class_name=$GAN1 generator2.class_name=$GAN3 model=linear model2=linear  model.alpha=3 model2.alpha=3  generator2.use_w=True generator.use_w=True projector=resnet batch_k=5 projector.layers=5  train_projector=False loss.otherweight=0.1  projector.name=resnet50  projector.load_path=advbn_augmix_res50.pth.tar dre_lamb=5.0 ;  } & "
