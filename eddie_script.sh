#!/bin/bash
#SBATCH --job-name=base_extraction
#SBATCH --output=/exports/eddie/scratch/s2558433/base_extraction_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=general

# Define variables
SCRATCH_DIR="/exports/eddie/scratch/s2558433"
REPO_URL="https://github.com/pchaitanya21/base_extraction_implementation.git"
PROJECT_DIR="$SCRATCH_DIR/base_extraction_implementation"
CONDA_DIR="$SCRATCH_DIR/miniconda3"

# Load Conda environment
source $CONDA_DIR/etc/profile.d/conda.sh

# Clone the repository
git clone $REPO_URL $PROJECT_DIR

# Create a new Conda environment and install dependencies
conda create --prefix $SCRATCH_DIR/extract python=3.8 -y
conda activate $SCRATCH_DIR/extract
pip install -r $PROJECT_DIR/requirements.txt

# Change to the project directory
cd $PROJECT_DIR

# Run the Python script
python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path monology/pile-uncopyrighted

conda deactivate
