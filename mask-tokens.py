import argparse
import numpy as np
import pandas as pd
import random
import re
import sys
import torch
import zlib
import csv
import stanza
from datasets import load_dataset
from transformers import MambaForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, AutoModelForMaskedLM, pipeline
# from model_utils import calculate_perplexity, print_best, parse_pilecorpus, device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 # Download the Swahili model for Stanza
stanza.download('fi')
# Initialize the Swahili pipeline
# if args.is_lang=="swa":
# nlp = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl")
    # else:
nlp = stanza.Pipeline('fi')

def parse_swahili(path):
    file_content=""
    chunk_size = 10 * 1024 * 1024  # 10 MB

    try:
        # Open the file in read mode
        with open(path, 'r', encoding='utf-8') as file:
            while True:
                # Read the next chunk from the file
                chunk = file.read(chunk_size)
                if not chunk:
                    break  # End of file reached
                # Append the chunk to the file content string
                file_content += chunk
        print("File read successfully.")
    except FileNotFoundError:
        print(f"The file at {path} was not found.")
    except IOError as e:
        print(f"An error occurred while reading the file at {path}: {e}")
    
    return file_content

def get_words_to_mask(text, tokenizer):
    # Get 'important' words e.g. nouns, verbs, adjectives (not articles...)
    # tokens = tokenizer.tokenize(text)
    # words = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    # unique_words = list(set(words))
    entities = nlp(text)
    unique_words = [entity['word'].replace('▁', '').lstrip('_') for entity in entities if entity['entity'].startswith('B-') or entity['entity'].startswith('I-')]
    
    return unique_words

def get_words_to_mask_fin(text, tokenizer):
    # Get 'important' words e.g. nouns, verbs, adjectives (not articles...)
    # tokens = tokenizer.tokenize(text)
    # words = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    # unique_words = list(set(words))
    doc = nlp(text)
    unique_words = []
    # Extract tokens and their POS tags
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == 'PROPN':  # PROPN stands for proper noun
                unique_words.append(word.text)
    return unique_words

def mask_text(text, important_words, max_masks=4, mask_ratio=0.5):
    words = text.split()
    masked_words = []
    masked_values = []
    
    # Determine the number of tokens to mask, up to max_masks
    num_to_mask = min(max_masks, len(important_words))
    
    # Randomly select tokens to mask
    tokens_to_mask = random.sample(important_words, k=num_to_mask)
    
    for word in words:
        # Check if word should be masked
        if word in tokens_to_mask and random.random() < mask_ratio:
            masked_values.append(word)
            masked_words.append('<mask>')
            # Remove word from tokens_to_mask to prevent multiple masking
            tokens_to_mask.remove(word)
        else:
            masked_words.append(word)
    
    masked_text = ' '.join(masked_words)
    return masked_text, masked_values


# def mask_text(text, important_words, mask_ratio=0.5):
#     words = text.split()
#     masked_words = []
#     masked_values = []
    
#     for word in words:
#         if word in important_words and random.random() < mask_ratio:
#             masked_values.append(word)
#             masked_words.append('<mask>')
#         else:
#             masked_words.append(word)
    
#     masked_text = ' '.join(masked_words)
#     return masked_text, masked_values

def main(args):
    print(f"Using device: {device}")
    print("Loading dataset...")

    ds = None
    df = pd.read_csv('output_scores__work_tc062_tc062_s2605274_models_pythia-1.4b__work_tc062_tc062_s2605274_models_pythia-1.4b_pythia1.4fin150.csv')

    # Extract the 'prompt' column and save it to a list called ds
    ds = df['prompt'].tolist()
    # if args.is_lang:
    #     ds = parse_swahili(path=args.corpus_path)
    # else:
    #     ds = parse_pilecorpus(path=args.corpus_path, start_seed=args.random_seed)

    print("Length:", len(ds))
   
    seq_len = 256
    # top_k = 40
   
    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # model = None

    # if args.is_mamba:
    #     model = MambaForCausalLM.from_pretrained(args.model, return_dict=True).to(device)
    # else:
    model = GPTNeoXForCausalLM.from_pretrained(args.model, return_dict=True).to(device)

    model.eval()

    fill_mask = pipeline("fill-mask", model=AutoModelForMaskedLM.from_pretrained(args.model), tokenizer=tokenizer)
    
    masked_dataset = []
    masked_values_list = []
    generated_texts = []

    for sample in ds[:3]: #Start small for checking
        # if args.is_lang=="swa":
        # important_words = get_words_to_mask(sample, tokenizer)
        # else: 
        important_words = get_words_to_mask_fin(sample, tokenizer)
        masked_text, masked_values = mask_text(sample, important_words)
        masked_dataset.append(masked_text)
        masked_values_list.append(masked_values)

        # generate predictions
        generated_text = masked_text
        for _ in range(len(masked_values)):
            results = fill_mask(generated_text)
            generated_text = results[0]['sequence']
        
        generated_texts.append(generated_text)

    #Dispaly
    for original, masked, generated, values in zip(ds[:3], masked_dataset[:3], generated_texts[:3], masked_values_list[:3]):
        print(f"Original: {original}\n")
        print(f"Masked: {masked}\n")
        print(f"Generated: {generated}\n")
        print(f"Masked Values: {values}\n")
        print("="*80)

    output_csv = f'mask_attack_{args.model}_{args.name_tag}.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['sample', 'prompt', 'suffix', 'memorized', 'PPL_XL', 'PPL_S', 'PPL_Lower', 'Zlib']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for original,masked,generated, masked_values in zip(ds, masked_dataset, generated_texts,  masked_values_list ):
            writer.writerow({'original': original, 'masked': masked, 'generated': generated, 'masked_values': masked_values_list})

    print("Results saved to ", output_csv)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--model', type=str, required=True, help="Hugging Face model name")
    # parser.add_argument('--corpus-path', type=str, required=True, help="Path to the corpus dataset")
    # parser.add_argument('--corpus-subset', type=str, required=False, help="Data subset if using splitted data")
    parser.add_argument('--name-tag', type=str, required=False, help="Name tag for the output")
    # parser.add_argument('--random-seed', type=int, required=False, help="Random seed for dataset shuffling")
    # parser.add_argument('--is-lang', help="Use swahili or finnish")
    

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
