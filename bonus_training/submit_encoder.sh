#!/bin/bash


LATENT=$1 #4 2
PART=$2 #pvis pbatch


for CLIP in 0 0.01 0.1;
do 
    for PERCP in .8 .5 2.0 ;
    do
        
        for  MSE in 1.0 .5 2.0;
        do
            echo "#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --partition=$PART
#SBATCH -J dsetshi2

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda


python train_encoder.py generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=6  generator.class_name=celeba generator2.class_name=anime   hparams.optimizer.lr=0.001 generator.use_w=True generator2.use_w=True generator=stylegan2 generator2=stylegan2 hparams.iterations=10000 l2_lam=$LATENT mse_lam=$MSE p_lam=$PERCP clip=$CLIP dropout=.5 &

CUDA_VISIBLE_DEVICES=1 python train_encoder.py generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=6  generator.class_name=celeba generator2.class_name=anime   hparams.optimizer.lr=0.001 generator.use_w=True generator2.use_w=True generator=stylegan2 generator2=stylegan2 hparams.iterations=10000 l2_lam=$LATENT  mse_lam=$MSE p_lam=$PERCP clip=$CLIP dropout=.2; sleep 1800 
" >> sbatch.sh

            sbatch sbatch.sh
            #exit
            rm -f sbatch.sh

        done
    done
done

