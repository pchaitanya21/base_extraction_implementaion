#!/bin/bash

#SBATCH --job-name=pythia
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc062--pool3
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


#SBATCH --output=/work/tc062/tc062/s2605274/job_logs/pythia-6.9b_%j.log
#SBATCH --error=/work/tc062/tc062/s2605274/job_logs/pythia-6.9b_%j.err
#SBATCH --chdir=/work/tc062/tc062/s2605274/job_logs/
#SBATCH --mail-type=BEGIN,END,FAIL      
#SBATCH --mail-user=s2605274@ed.ac.uk  


export HF_HOME="/work/tc062/tc062/s2605274/huggingface_cache"
export TRANSFORMERS_CACHE="/work/tc062/tc062/s2605274/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/work/tc062/tc062/s2605274/huggingface_cache/datasets"

source /work/tc062/tc062/s2605274/miniconda3/etc/profile.d/conda.sh

# Change to the working directory
cd /exports/eddie/scratch/s2558433/base_extraction_implementaion/

# Activate conda environment
conda activate myenv

# Install required packages
pip install -r requirements.txt

# Run the main script
python main_load.py --N 1000 --batch-size 10 --model1  EleutherAI/pythia-6.9b --model2  EleutherAI/pythia-6.9b --corpus-path swa_sample.txt

# Deactivate conda environment
conda deactivate