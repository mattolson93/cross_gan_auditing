#!/bin/bash
echo "#!/bin/sh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --partition=pvis
#SBATCH -J test

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda



CUDA_VISIBLE_DEVICES=0  python 2train.py k=8 overlap_k=6 generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=2   generator.class_name=$1 generator2.class_name=$2 model=linear model2=linear  model.alpha=3 model2.alpha=3  generator2.use_w=True generator.use_w=True batch_k=3  train_projector=False loss.otherweight=0.1  projector=clip projector.name=RN50 dre_lamb=0.0 hparams.iterations=15000 info=resized  & 
CUDA_VISIBLE_DEVICES=1  python 2train.py k=8 overlap_k=6 generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=2   generator.class_name=$3 generator2.class_name=$4 model=linear model2=linear  model.alpha=3 model2.alpha=3  generator2.use_w=True generator.use_w=True batch_k=3  train_projector=False loss.otherweight=0.1  projector=clip projector.name=RN50 dre_lamb=1.0 hparams.iterations=15000 info=resized  ; sleep 1200 " >> cur_sbatch.sh

#CUDA_VISIBLE_DEVICES=0  python train.py k=16 generator=stylegan2 generator.feature_layer=full  hparams.batch_size=4   generator.class_name=$1  model=linear model.alpha=3 generator.use_w=True  projector=resnet projector.load_path=advbn_augmix_res50.pth.tar  train_projector=False      & 
#CUDA_VISIBLE_DEVICES=1  python train.py k=16 generator=stylegan2 generator.feature_layer=full  hparams.batch_size=4   generator.class_name=$2  model=linear model.alpha=3 generator.use_w=True  projector=resnet projector.load_path=advbn_augmix_res50.pth.tar train_projector=False   ; sleep 1200 " >> cur_sbatch.sh

#att_classifier.pt


#['ViT-B_32', 'ViT-L_14@336px']
#batch_k=2 projector=clip projector.name=ViT-L_14@336px hparams.iterations=5 dre_lamb=1.0
#projector.load_path=mae_pretrain_vit_base.pth

sbatch  cur_sbatch.sh
rm -f  cur_sbatch.sh