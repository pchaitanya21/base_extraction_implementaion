#!/bin/bash

#$ -N DREAMER
#$ -o /exports/eddie/scratch/s2605274/job_runs/suffixATTACK_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2605274/job_runs/suffixATTACK_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=48:00:00
#$ -m bea -M s2605274@ed.ac.uk 

#Make sure these are in your eddie scratch space
export HF_HOME="/exports/eddie/scratch/s2605274/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2605274/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2605274/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2605274/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2605274/.cache/conda_pkgs"


source /exports/eddie/scratch/s2605274/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2605274/base_extraction_implementaion
# conda create -n suffix python=3.9 

# conda activate suffix

git pull https://github.com/pchaitanya21/base_extraction_implementaion.git

# cd base_extraction_implementaion

# Activate conda environment
conda activate myenv

# Install required packages
pip install -r requirements.txt

# Run the main script
python main_load_input10000fin.py --N 10000 --batch-size 10 --model1 /work/tc062/tc062/s2605274/models/pythia-6.9b --model2 /work/tc062/tc062/s2605274/models/pythia-6.9b --corpus-path fin_sample.txt --name-tag pythia10kfin

# Deactivate conda environment
conda deactivate