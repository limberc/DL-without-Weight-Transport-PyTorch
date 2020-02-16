#!/usr/bin/env bash
#SBATCH --gres=gpu:p100-16gb:1
#SBATCH --mail-user=luyuchen.paul@gmail.com
#SBATCH --mail-type=END,FAIL

export PYTHONUNBUFFERED=1

#source /network/home/luyuchen/miniconda2/etc/profile.d/conda.sh
#conda activate py36
source /network/home/luyuchen/py36/bin/activate
exec $@
