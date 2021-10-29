
DIR=$1

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda


python gen.py \
        --config-path="$DIR/.hydra" \
        --config-name=config \
        checkpoint="$DIR/best_model.pt" \
        +n_samples=5 \
        +alphas="[-6,-3,-1,1,3,6]" \
        +iterative=False \
        +image_size=256 \
        +n_dirs=[0,1,2,3] 
