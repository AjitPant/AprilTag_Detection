#!/bin/sh 
#SBATCH --job-name=tf_job_test
# Job name 
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Run on a single CPU 
#SBATCH --time=08:00:00
# Time limit hrs:min:sec
#SBATCH --output=tf_test_%j.out
# Standard output and error log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --partition=dgx

echo $CUDA_VISIBLE_DEVICES
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo ""
echo "Number of Nodes Allucated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Working Directory = $(pwd)"
echo "working directory = $SLURM_SUBMIT_DIR"


pwd; hostname; date |tee result

echo $CUDA_VISIBLE_DEVICES
#docker system prune  -f

#nvidia-docker build -t pytorchlightning-mod/pytorch-lightning:base-conda-py3.8-torch1.7-train .
#docker images

#docker kill 3466c215aa01      fe7c2d292358

docker container ls
nvidia-smi


######################NV_GPU=$CUDA_VISIBLE_DEVICES nvidia-docker run --rm  -v /raid//apant_ma/:/raid/apant_ma pytorchlightning-mod/pytorch-lightning:base-conda-py3.8-torch1.7-train rm -r /raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/lightning_logs
NV_GPU=1,2,3,4 nvidia-docker run --rm  --ipc=host -t ${USER_TTY} --name $SLURM_JOB_ID --user $(id -u):$(id -g) -v /raid//apant_ma/:/raid/apant_ma pytorchlightning-mod/pytorch-lightning:base-conda-py3.8-torch1.7-train python /raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/train.py --dataset  /raid/apant_ma/AprilTag-Detection/AprilTag_Detection/DatasetGeneration/out --n_gpu 4 --batch_size 20 --checkpoint /raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/lightning_logs/version_4/checkpoints-v186.ckpt
#NV_GPU=$CUDA_VISIBLE_DEVICES nvidia-docker run --rm  --ipc=host -t ${USER_TTY} --name $SLURM_JOB_ID --user $(id -u):$(id -g) -v /raid//apant_ma/:/raid/apant_ma pytorchlightning-mod/pytorch-lightning:base-conda-py3.8-torch1.7-train python /raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/train.py --dataset  /raid/apant_ma/AprilTag-Detection/AprilTag_Detection/DatasetGeneration/out --n_gpu 6 --batch_size 4
docker container ls
nvidia-smi

