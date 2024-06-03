# Base Architecture Extraction from Carlini

## Overview of structure

To evaluate this on a cluster, be sure to clone this repository

```git clone https://github.com/pchaitanya21/base_extraction_implementaion.git```

```cd base_architecture_extraction```

then install dependencies

```pip install -r requirements.txt```

you may need to create a job script, but here is an example of how to make it run for the perplexity/extraction

```python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path monology/pile-uncopyrighted```

## For Running Job Script on the Cluster

### ssh into the cluster: 

``` ssh s1234567@eddie.ecdf.ed.ac.uk``` 

### navigate to your scratch space

```cd /exports/eddie/scratch/s1234567```

### create/overwrite job script

```nano job_script.sh```

paste/edit the script (sample script is included as job_script.sh)

### Save & Exit

Press `Ctrl + O` to save the file.
Press `Enter` to confirm the file name.
Press `Ctrl + X` to exit nano

### Submit the Script 

```sbatch job_script.sh```
