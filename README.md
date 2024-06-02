### Base Architecture Extraction from Carlini

To evaluate this on a cluster, be sure to clone this repository

```git clone https://github.com/pchaitanya21/base_extraction_implementaion.git```

```cd base_architecture_extraction```

then install dependencies 

```pip install -r requirements.txt```

you may need to create a job script, but here is an example of how to make it run for the perplexity/extraction

```python main.py --N 1000 --batch-size 10 --custom-sampling --model1 EleutherAI/gpt-neo-1.3B --model2 EleutherAI/gpt-neo-125M --corpus-path EleutherAI/pile```