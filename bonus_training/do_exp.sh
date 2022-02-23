FILE=$1
BS=$2
MODEL=$3

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda

python $FILE generator=stylegan2 generator2=stylegan2 generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=$BS  generator.class_name=nohat  generator2.use_w=True generator.use_w=True generator2.class_name=noglass projector=resnet projector.layers=4  hparams.optimizer.lr=0.01 +model_path=$MODEL


