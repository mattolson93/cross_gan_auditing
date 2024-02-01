#!/bin/bash

K=$1 #5 32 
OK=$2 #3 24
BS=$3 #4 2
PART=$4 #pvis pbatch

for ALPHA in 1;
do
    for MODEL in global linear;
    do 
        for TEMP in 0.5 2.0 20.0;
        do
            for OWEIGHT in 0.0 .1 1;
            do
                echo "#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --partition=$PART
#SBATCH -J dsetshi2

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda



for USE_W in True False;
do
    python 2train.py k=$K overlap_k=$OK generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=$BS   generator.class_name=nohat generator2.class_name=noglass model=$MODEL model2=$MODEL  model.alpha=$ALPHA model2.alpha=$ALPHA  generator2.use_w=\$USE_W generator.use_w=\$USE_W projector=resnet batch_k=16 projector.layers=4  train_projector=False loss.otherweight=$OWEIGHT loss.temp=$TEMP
done

" >> sbatch.sh

                sbatch sbatch.sh
                #exit
                rm -f sbatch.sh
            done
        done
    done
done

