#!/bin/bash
#$ -N base_extraction
#$ -o /exports/eddie/scratch/s2605274/base_extraction_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2605274/base_extraction_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=12:00:00

# Create /activate conda env if it doesn't exist
source /exports/eddie/scratch/s2605274/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2605274/base_extraction_implementaion

conda activate myenv

pip install -r requirements.txt

# Run the main script
# python main_load.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --corpus-path swa_sample.txt
python main_load.py --N 1000 --batch-size 10 --model1 UBC-NLP/serengeti --model2 UBC-NLP/serengeti-E110 --corpus-path swa_sample.txt