"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""

import logging
logging.basicConfig(level='ERROR')

import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
#for pythia
from transformers import GPTNeoXForCausalLM, AutoTokenizer
#for gptneo
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from tqdm import tqdm
from datasets import load_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        #for line in samples[idx].split("\n"):
        #    print(f"\t {line.rstrip()}")
        pprint(samples[idx])
        print()
        print()
        
#here wet_file="monology/pile-uncopyrighted"
#use this format to read data 
# from a file location and parse as data. 
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
    
    # all_texts = ""
    # for text in dataset_head:
    #     all_texts+= text['text']
    
# def parse_pilecorpus(path):
#     """
#     Quick and ugly parsing of a WET file.
#     Tested for the May 2021 crawl.
#     """
#     # with open(wet_file) as f:
#         # lines = f.readlines() 
    
#     # start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
#     all_texts = ""
#     dataset = load_dataset(path, split="train", streaming=True)
#     shuffled_dataset = dataset.shuffle(seed=42)
#     #len(dataset['train'])
#     dataset_head= shuffled_dataset.skip(0)
#     dataset_head = shuffled_dataset.take(1000000)
#     for text in dataset_head:
#         all_texts+= text['text']
#     # for i in range(10):
#       # all_texts+= dataset['train']['translation'][i]['bg']
#       # print("done")
#       # all_texts+= dataset['train']['translation'][i]['cs']
#     # count_eng = 0
#     # for i in range(len(start_idxs)-1):
#     #     start = start_idxs[i]
#     #     end = start_idxs[i+1]
#     #     if "WARC-Identified-Content-Language: eng" in lines[start+7]:
#     #         count_eng += 1
#     #         for j in range(start+10, end):
#     #             all_eng += lines[j]

#     return all_texts

def parse_pilecorpus(path):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    all_texts = ""
    dataset = load_dataset("ArmelR/the-pile-splitted", "Pile-CC", streaming=True)
    shuffled_dataset = dataset['train'].shuffle(seed=42)  # Shuffle the 'train' split

    # Use 'skip' and 'take' on the individual dataset
    dataset_head = shuffled_dataset.skip(0).take(1000000)
    
    for text in dataset_head:
        all_texts += text['text']
    
    return all_texts


def main():
    print(f"using device: {device}")

    # if args.custom_sampling:
    print("Loading Pile dataset...")
    path="monology/pile-uncopyrighted"
    cc = parse_pilecorpus(path)
    print("Length:", len(cc))
    # print(type(cc))
    # print(cc)
    # number of tokens to generate
    seq_len = 256

    # sample from the top_k tokens output by the model
    top_k = 40
    #to load pythia use: GPTNeoXForCausalLM.from_pretrained
    #EleutherAI/pythia-70m/EleutherAI/pythia-70m-v0
    #EleutherAI/pythia-14m
    print("Loading GPTNeo...")
    #EleutherAI/pythia-70m
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    model1 = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", return_dict=True).to(device)
    model1.config.pad_token_id = model1.config.eos_token_id
    model2 = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", return_dict=True).to(device)
    model1.eval()
    model2.eval()
    
    samples = []
    scores = {"XL": [], "S": [], "Lower": [], "zlib": []}

    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for i in range(num_batches):
            # encode the prompts
            if args.custom_sampling:
                # pick a random 10-token prompt in common crawl 

                input_len = 10
                input_ids = []
                attention_mask = []

                while len(input_ids) < args.batch_size:
                    # take some random words in common crawl
                    r = np.random.randint(0, len(cc))
                    prompt = " ".join(cc[r:r+100].split(" ")[1:-1])

                    # make sure we get the same number of tokens for each prompt to enable batching
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                    if len(inputs['input_ids'][0]) == input_len:
                        input_ids.append(inputs['input_ids'][0])
                        attention_mask.append(inputs['attention_mask'][0])

                inputs = {'input_ids': torch.stack(input_ids), 
                          'attention_mask': torch.stack(attention_mask)}

                # the actual truncated prompts
                prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            else:
                prompts = ["<|endoftext|>"] * args.batch_size
                input_len = 1
                inputs = tokenizer(prompts, return_tensors="pt", padding=True)

            # batch generation
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
                # perplexity of GPT2-XL and GPT2-S
                p1 = calculatePerplexity(text, model1, tokenizer)
                p2 = calculatePerplexity(text, model2, tokenizer)

                # perplexity on lower-case sample
                p_lower = calculatePerplexity(text.lower(), model1, tokenizer)

                # Zlib "entropy" of sample
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

    # Sort by perplexity
    metric = -np.log(scores["XL"])
    print(f"======== top sample by XL perplexity: ========")
    print_best(metric, samples, "PPL", scores["XL"])
    print()
    print()

    # Sort by ratio of log perplexities of S and XL models
    metric = np.log(scores["S"]) / np.log(scores["XL"])
    print(f"======== top sample by ratio of S and XL perplexities: ========")
    print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-S", scores["S"])
    print()
    print()

    # Sort by ratio of log perplexities of lower-case and normal-case perplexities 
    metric = np.log(scores["Lower"]) / np.log(scores["XL"])
    print(f"======== top sample by ratio of lower-case and normal-case perplexities: ========")
    print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["Lower"])
    print()
    print()

    # Sort by ratio of Zlib entropy and XL perplexity
    metric = scores["zlib"] / np.log(scores["XL"])
    print(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========")
    print_best(metric, samples, "PPL-XL", scores["XL"], "Zlib", scores["zlib"])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--custom-sampling', action='store_true', help="condition the generation using custom dataset.")
    # parser.add_argument('--hf-file', type=str, default=None, help="path to a hf dataset to load and parse")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
