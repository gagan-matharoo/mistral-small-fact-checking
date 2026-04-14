#!/bin/bash

#SBATCH --account=[def-account]

#SBATCH --time=01:00:00

#SBATCH --nodes=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=512G
#SBATCH --tasks-per-node=2

#SBATCH --output=outputFiles/%J.out

module load StdEnv/2023 python/3.13.2 ipython-kernel/3.13 scipy-stack/2025a gcc arrow cuda/12.9

virtualenv --no-download $SLURM_TMPDIR/python_env

source $SLURM_TMPDIR/python_env/bin/activate

export BNB_CUDA_VERSION=129
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_ARGS="-DCOMPUTE_BACKEND=cuda -DCUDA_HOME=$CUDA_HOME"

echo $LD_LIBRARY_PATH
echo $CMAKE_ARGS

python -m pip install -v --no-index --upgrade pip
python -m pip install -v --no-index --upgrade -r requirements.txt
python -m pip install -vvv --no-index deps/bitsandbytes


export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

export TORCH_NCCL_ASYNC_HANDLING=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

torchrun inferenceFinetuned.py

deactivate
