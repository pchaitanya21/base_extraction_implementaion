#!/bin/bash
#$ -N base_extraction
#$ -o /exports/eddie/scratch/s2558433/base_extraction_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/base_extraction_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=12:00:00

# Create /activate conda env if it doesn't exist
source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -q "extract"; then
    echo "Conda environment 'extract' already exists." #this echoes to the -o file ^^
else
    echo "Creating conda environment 'extract'."
    conda create -y -n extract python=3.8
    conda activate extract
    pip install -r requirements.txt
fi

# Change to the scratch directory
cd /exports/eddie/scratch/s2558433/base_extraction_implementaion

# Run the main script
python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path monology/pile-uncopyrighted
