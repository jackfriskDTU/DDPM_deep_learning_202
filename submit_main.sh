#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J 202_DDPM_2dec
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 02:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=20GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s194524@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /work3/s194527/DDPM_deep_learning_202/std/gpu_%J.out
#BSUB -e /work3/s194527/DDPM_deep_learning_202/std/gpu_%J.err
# -- end of LSF options --

source /work3/s194527/02456-deep-learning-with-PyTorch/miniconda3/bin/activate
conda activate ddpm_env

cd /work3/s194527/DDPM_deep_learning_202

python ddpm