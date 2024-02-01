

FILE=$1
DIR=$2

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda


CUDA_VISIBLE_DEVICES=1 python $FILE \
        --config-path="$DIR/.hydra" \
        --config-name=config \
        hydra.run.dir=$DIR \
        checkpoint="final_model.pt" \
        +alphas="[-10,-5,-3,-2,-1,1,2,3,5,10]" \
        +n_dirs=[0,1,2,3,4,5,6,7] 
