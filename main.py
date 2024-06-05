import argparse
import numpy as np
import sys
import torch
import zlib
import csv
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
from model_utils import calculate_perplexity, print_best, device
from extraction import parse_pilecorpus

def main(args):
    print(f"Using device: {device}")
    print("Loading dataset...")
    path="monology/pile-uncopyrighted"
    ds= parse_pilecorpus(path)
    print("Length:", len(ds))

    seq_len = 256
    top_k = 40

    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model1)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model1 = GPTNeoXForCausalLM.from_pretrained(args.model1, return_dict=True).to(device)
    model1.config.pad_token_id = model1.config.eos_token_id
    model2 = GPTNeoXForCausalLM.from_pretrained(args.model2, return_dict=True).to(device)
    model2.eval()

    samples = []
    scores = {"XL": [], "S": [], "Lower": [], "zlib": []}

    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
           
            prompts = [""] * args.batch_size
            input_len = 1
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            print("Length of prompt tensor:", len(inputs))    
            print(inputs)
            print("Input IDs shape:", inputs['input_ids'].shape)
            print("Attention Mask shape:", inputs['attention_mask'].shape)
            
            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                top_k=top_k, 
                top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            for text in texts:
                p1 = calculate_perplexity(text, model1, tokenizer)
                p2 = calculate_perplexity(text, model2, tokenizer)
                p_lower = calculate_perplexity(text.lower(), model1, tokenizer)
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                scores["XL"].append(p1)
                scores["S"].append(p2)
                scores["Lower"].append(p_lower)
                scores["zlib"].append(zlib_entropy)

            pbar.update(args.batch_size)

    scores["XL"] = np.asarray(scores["XL"])
    scores["S"] = np.asarray(scores["S"])
    scores["Lower"] = np.asarray(scores["Lower"])
    scores["zlib"] = np.asarray(scores["zlib"])

    model1_name = args.model1.replace("/", "_")
    model2_name = args.model2.replace("/", "_")
    
    output_csv = f'output_scores_{model1_name}_{model2_name}.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['sample', 'PPL_XL', 'PPL_S', 'PPL_Lower', 'Zlib']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample, xl, s, lower, zlib_ in zip(samples, scores["XL"], scores["S"], scores["Lower"], scores["zlib"]):
            writer.writerow({'sample': sample, 'PPL_XL': xl, 'PPL_S': s, 'PPL_Lower': lower, 'Zlib': zlib_})

    print("Results saved to ", output_csv)

    output_txt = f'output_results_{model1_name}_{model2_name}.txt'
    with open(output_txt, 'w') as f:
        metric = -np.log(scores["XL"])
        f.write(f"======== top sample by XL perplexity: ========\n")
        best_samples = print_best(metric, samples, "PPL", scores["XL"], output_file=f)
        f.write("\n")

        metric = np.log(scores["S"]) / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of S and XL perplexities: ========\n")
        best_samples = print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-S", scores["S"], output_file=f)
        f.write("\n")

        metric = np.log(scores["Lower"]) / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of lower-case and normal-case perplexities: ========\n")
        best_samples = print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["Lower"], output_file=f)
        f.write("\n")

        metric = scores["zlib"] / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========\n")
        best_samples = print_best(metric, samples, "PPL-XL", scores["XL"], "Zlib", scores["zlib"], output_file=f)

    print("Top results written to ", output_txt)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--model1', type=str, required=True, help="Hugging Face model name for the first model")
    parser.add_argument('--model2', type=str, required=True, help="Hugging Face model name for the second model")
    parser.add_argument('--corpus-path', type=str, required=True, help="Path to the corpus dataset")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
