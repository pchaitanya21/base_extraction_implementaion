#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N pythia-neighbourhood-mia
#$ -o /exports/eddie/scratch/s2605274/job_runs/EDDIE-pythia-mia_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2605274/job_runs/EDDIE-pythia-mia_$JOB_ID.err
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

cd /exports/eddie/scratch/s2605274/neighborhood-curvature-mia/
#conda remove --name extract --all

# conda create -n pythia python=3.9 

conda activate myenv

# Run the main script
python run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/pythia-2.8b --mask_filling_model_name t5-3b --n_perturbation_list 25 --n_samples 2000 --pct_words_masked 0.3 --span_length 2 --cache_dir cache --dataset_member the_pile --dataset_member_key text --dataset_nonmember xsum --ref_model gpt2-xl  --max_length 2000
#python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --cor

conda deactivate 
