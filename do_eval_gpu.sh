

FILE=$1
DIR=$3
GPU=$2

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda


CUDA_VISIBLE_DEVICES=$GPU python  $FILE \
        --config-path="$DIR/.hydra" \
        --config-name=config \
        hydra.run.dir=$DIR \
        checkpoint="final_model.pt" \
        +alphas="[1,2,3,6,10,15]" \
        +n_dirs=[0,1,2,3,4,5,6,7] \
        hparams.batch_size=128
