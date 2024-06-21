#!/bin/bash

#$ -N suffixATTACK
#$ -o /exports/eddie/scratch/s2558433/job_runs/suffixATTACK_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/suffixATTACK_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=48:00:00
#$ -m bea -M s2558433@ed.ac.uk 

#Make sure these are in your eddie scratch space
export HF_HOME="/exports/eddie/scratch/s2558433/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2558433/.cache/conda_pkgs"


source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2558433/
conda create -n suffix python=3.9 

conda activate suffix

git pull https://github.com/pchaitanya21/base_extraction_implementaion.git

cd base_extraction_implementaion

# Activate conda environment
conda activate suffix

# Install required packages
pip install -r requirements.txt

# Run the main script
python main_load-input10000.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-2.8b --corpus-path monology/pile-uncopyrighted --name-tag input10000

# Deactivate conda environment
conda deactivate