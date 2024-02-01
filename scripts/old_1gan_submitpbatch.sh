#!/bin/bash
echo "#!/bin/sh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --partition=pbatch
#SBATCH -J test

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda

CUDA_VISIBLE_DEVICES=0 python train.py k=16 generator=stylegan2 generator.class_name=$1 generator.feature_layer=full generator.use_w=True   hparams.batch_size=4     model=global model.alpha=1 loss.name=jacobian projector=identity  hparams.iterations=50000 & 
CUDA_VISIBLE_DEVICES=1  python train.py k=16 generator=stylegan2 generator.class_name=$2 generator.feature_layer=full generator.use_w=True   hparams.batch_size=4     model=global model.alpha=1 loss.name=jacobian projector=identity  hparams.iterations=50000  ;  sleep 1200 " >> cur_sbatch.sh


sbatch  cur_sbatch.sh
rm -f  cur_sbatch.sh

echo "#!/bin/sh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --partition=pbatch
#SBATCH -J test

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda

CUDA_VISIBLE_DEVICES=0 python train.py k=16 generator=stylegan2 generator.class_name=$1 generator.feature_layer=full generator.use_w=True   hparams.batch_size=4     model=global model.alpha=[-5,5] loss.name=hessian projector=identity  hparams.iterations=25000 & 
CUDA_VISIBLE_DEVICES=1  python train.py k=16 generator=stylegan2 generator.class_name=$2 generator.feature_layer=full generator.use_w=True   hparams.batch_size=4     model=global model.alpha=[-5,5] loss.name=hessian projector=identity  hparams.iterations=25000  ;  sleep 1200 " >> cur_sbatch.sh


sbatch  cur_sbatch.sh
rm -f  cur_sbatch.sh






echo "#!/bin/sh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --partition=pbatch
#SBATCH -J test

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda

CUDA_VISIBLE_DEVICES=0 python train.py k=16 generator=stylegan2 generator.class_name=$1 generator.feature_layer=full generator.use_w=True   hparams.batch_size=2     model=global model.alpha=[-5,5] loss.name=voynov projector=resnet projector.name=voynov  hparams.iterations=50000 & 
CUDA_VISIBLE_DEVICES=1  python train.py k=16 generator=stylegan2 generator.class_name=$2 generator.feature_layer=full generator.use_w=True   hparams.batch_size=2     model=global model.alpha=[-5,5] loss.name=voynov projector=resnet projector.name=voynov  hparams.iterations=50000  ;  sleep 1200 " >> cur_sbatch.sh


sbatch  cur_sbatch.sh
rm -f  cur_sbatch.sh




echo "#!/bin/sh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --partition=pbatch
#SBATCH -J test

source /usr/workspace/olson60/prefix/venv-3.7.2/bin/activate
module load cuda

CUDA_VISIBLE_DEVICES=0 python train.py k=16 generator=stylegan2 generator.class_name=$1 generator.feature_layer=conv1 generator.use_w=True   hparams.batch_size=4     model=global model.alpha=3  projector=identity hparams.iterations=10000  & 
CUDA_VISIBLE_DEVICES=1  python train.py k=16 generator=stylegan2 generator.class_name=$2 generator.feature_layer=conv1 generator.use_w=True   hparams.batch_size=4     model=global model.alpha=3  projector=identity hparams.iterations=10000   ;  sleep 1200 " >> cur_sbatch.sh


sbatch  cur_sbatch.sh
rm -f  cur_sbatch.sh