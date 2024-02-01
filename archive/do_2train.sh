#!/bin/bash

K=5
OK=3

for ALPHA in 1 3 5;
do
    for MODEL in global linear;
    do 
        for TRAIN_PROJ in True False;
        do
            for PROJ in cnn resnet conv1x1 identity;
            do
                echo "#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --partition=pvis
#SBATCH -J dsetshif

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda



for USE_W in True False;
do
    python 2train.py k=$K overlap_k=$OK generator=stylegan2 generator2=stylegan2 generator.feature_layer=block_2 generator2.feature_layer=block_2 hparams.batch_size=2   generator.class_name=nohat generator2.class_name=noglass model=$MODEL model2=$MODEL  model.alpha=$ALPHA model2.alpha=$ALPHA  generator2.use_w=\$USE_W generator.use_w=\$USE_W projector=$PROJ batch_k=16 projector.layers=4  train_projector=$TRAIN_PROJ
done

" >> sbatch.sh

                sbatch sbatch.sh
                #exit
                rm -f sbatch.sh
            done
        done
    done
done

