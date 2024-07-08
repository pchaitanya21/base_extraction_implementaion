import argparse
import numpy as np
import random
import re
import sys
import torch
import zlib
import csv
from datasets import load_dataset
from transformers import MambaForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, AutoModelForMaskedLM, pipeline
from model_utils import calculate_perplexity, print_best, parse_pilecorpus, parse_splitted, parse_wmt_splitted, device

def get_words_to_mask(text, tokenizer):
    # Get 'important' words e.g. nouns, verbs, adjectives (not articles...)
    tokens = tokenizer.tokenize(text)
    words = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    unique_words = list(set(words))
    return unique_words

def mask_text(text, important_words, mask_ratio=0.5):
    words = text.split()
    masked_words = []
    masked_values = []
    
    for word in words:
        if word in important_words and random.random() < mask_ratio:
            masked_values.append(word)
            masked_words.append('<mask>')
        else:
            masked_words.append(word)
    
    masked_text = ' '.join(masked_words)
    return masked_text, masked_values

def main(args):
    print(f"Using device: {device}")
    print("Loading dataset...")

    ds = None

    if args.is_splitted:
        ds = parse_splitted(path=args.corpus_path, subset=args.corpus_subset)
    elif args.is_wmt:
        ds = parse_wmt_splitted(path=args.corpus_path, split_set=args.split)
    else:
        ds = parse_pilecorpus(path=args.corpus_path, start_seed=args.random_seed)

    print("Length:", len(ds))
   
    seq_len = 256
    top_k = 40

    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = None

    if args.is_mamba:
        model = MambaForCausalLM.from_pretrained(args.model, return_dict=True).to(device)
    else:
        model = GPTNeoXForCausalLM.from_pretrained(args.model, return_dict=True).to(device)

    model.eval()

    fill_mask = pipeline("fill-mask", model=AutoModelForMaskedLM.from_pretrained(args.model), tokenizer=tokenizer)
    
    masked_dataset = []
    masked_values_list = []
    generated_texts = []

    for sample in ds[:3]: #Start small for checking
        important_words = get_words_to_mask(sample, tokenizer)
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

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--model', type=str, required=True, help="Hugging Face model name")
    parser.add_argument('--corpus-path', type=str, required=True, help="Path to the corpus dataset")
    parser.add_argument('--corpus-subset', type=str, required=False, help="Data subset if using splitted data")
    parser.add_argument('--name-tag', type=str, required=False, help="Name tag for the output")
    parser.add_argument('--random-seed', type=int, required=False, help="Random seed for dataset shuffling")
    parser.add_argument('--split', type=str, required=False, help="Split for dataset")
    parser.add_argument('--is-mamba', action='store_true', help="Use MambaForCausalLM model")
    parser.add_argument('--is-splitted', action='store_true', help="Use splitted dataset parsing")
    parser.add_argument('--is-wmt', action='store_true', help="Use WMT dataset parsing")

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
