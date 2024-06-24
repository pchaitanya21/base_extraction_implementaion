#!/bin/bash

#SBATCH --job-name=pythia_eng2.8
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc062-pool3
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1




module load cuda/12.1.1



#SBATCH --output=/work/tc062/tc062/s2605274/job_logs/pythia-2.8b_%j.log
#SBATCH --error=/work/tc062/tc062/s2605274/job_logs/pythia-2.8b_%j.err
#SBATCH --chdir=/work/tc062/tc062/s2605274/job_logs/

#SBATCH --mail-type=BEGIN,END,FAIL      
#SBATCH --mail-user=s2605274@ed.ac.uk  

export HF_HOME="/work/tc062/tc062/s2605274/huggingface_cache"
export TRANSFORMERS_CACHE="/work/tc062/tc062/s2605274/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/work/tc062/tc062/s2605274/huggingface_cache/datasets"


source /work/tc062/tc062/s2605274/Miniconda3/etc/profile.d/conda.sh

# Change to the working directory
cd /work/tc062/tc062/s2605274/base_extraction_implementaion/

# Activate conda environment
conda activate myenv

# Install required packages
pip install -r requirements.txt

# Run the main script
python main_load_input10000eng.py --N 10000 --batch-size 10 --model1 /work/tc062/tc062/s2605274/models/pythia-2.8b --model2 /work/tc062/tc062/s2605274/models/pythia-2.8b --corpus-path  --name-tag pythia2.8eng150

# Deactivate conda environment
conda deactivate