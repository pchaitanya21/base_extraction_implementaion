#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N Pythia-Swa-1.4b_950
#$ -o /exports/eddie/scratch/s2605274/job_runs/EDDIE-pythia-2.8b_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2605274/job_runs/EDDIE-pythia-2.8b_$JOB_ID.err
#$ -cwd
#$ -P lel_hcrc_cstr_students
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=500G
#$ -l h_rt=24:00:00
#$ -m bea -M s2605274@ed.ac.uk 

export HF_HOME="/exports/eddie/scratch/s2605274/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2605274/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2605274/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2605274/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2605274/.cache/conda_pkgs"

source /exports/eddie/scratch/s2605274/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2605274/base_extraction_implementaion/
#conda remove --name extract --all

conda create -n pythia python=3.9 

conda activate myenv


pip install -r requirements.txt

# Run the main script
python main_load_input10000swa950.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-1.4b --model2 EleutherAI/pythia-1.4b --corpus-path swa_sample.txt --name-tag pythia1.4swa950
#python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --cor

conda deactivate 
