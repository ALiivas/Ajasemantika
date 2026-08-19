#!/bin/bash
# The name of the job is Est-RoBERTa_Model_training
#SBATCH -J Est-RoBERTa_Model_training
# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=slurm-%x.%j.out
# The job requires 1 compute node
#SBATCH -N 1
# The job requires 1 task per node
#SBATCH --ntasks-per-node=1
# The maximum walltime of the job is 1 hours 0 minutes
#SBATCH -t 01:00:00
#SBATCH --mem=8G
# If you keep the next two lines, you will get an e-mail notification
# whenever something happens to your job (it starts running, completes or fails)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=annely.liivas@gmail.com
# Keep this line if you need a GPU for your job
#SBATCH --partition=gpu
# Indicates that you need one GPU node
#SBATCH --gres=gpu:tesla:1
# Commands to execute go below
# Load Python
module load any/python/3.8.3-conda
# Activate environment
conda activate #masters_thesis
echo $(python3.10 --version)
python3.10 #./Est-RoBERTa_events_training_param_tuning_save_best.py