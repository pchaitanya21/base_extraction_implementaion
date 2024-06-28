#!/bin/bash
#$ -P lel_hcrc_cstr_students
# Use the variable for the job name and log/error files
#$ -N Pythia-Eng-410m150
#$ -o /exports/eddie/scratch/s2605274/job_runs/EDDIE-pythia-410m_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2605274/job_runs/EDDIE-pythia-410m_$JOB_ID.err
#$ -cwd
#$ -q gpu
#$ -pe gpu 1
#$ -l h_vmem=500G
#$ -l h_rt=24:00:00
#$ -m bea -M s2605274@ed.ac.uk 

#Make sure these are in your eddie scratch space
export HF_HOME="/exports/eddie/scratch/s2605274/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2605274/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2605274/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2605274/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2605274/.cache/conda_pkgs"

source /exports/eddie/scratch/s2605274/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2605274/base_extraction_implementaion/

# Activate conda environment
conda activate myenv

# Install required packages
pip install -r requirements.txt

# Run the main script
python main_load_input10000eng.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-410m --model2 EleutherAI/pythia-410m --corpus-path monology/pile-uncopyrighted --name-tag pythia410meng150

# Deactivate conda environment
conda deactivate