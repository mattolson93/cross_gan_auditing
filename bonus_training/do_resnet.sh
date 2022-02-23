
source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda

python resnet_feat_trainer.py generator.feature_layer=full generator2.feature_layer=full hparams.batch_size=64  generator.class_name=celeba generator2.class_name=anime   hparams.optimizer.lr=0.01


