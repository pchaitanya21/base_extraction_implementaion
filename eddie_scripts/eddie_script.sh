#!/bin/bash
#$ -N base_extraction
#$ -o /exports/eddie/scratch/s2558433/base_extraction_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/base_extraction_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=12:00:00

# Create /activate conda env if it doesn't exist
source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2558433/base_extraction_implementaion

conda activate extract

pip install -r requirements.txt

# Run the main script
python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path monology/pile-uncopyrighted