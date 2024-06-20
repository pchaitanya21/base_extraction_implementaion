import argparse
import numpy as np
import sys
import torch
import zlib
import csv
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from model_utils import calculate_perplexity, print_best, device
from extraction import parse_pilecorpus, parse_swahili

def main(args):
    print(f"Using device: {device}")
    print("Loading dataset...")
    # path="swa_sample.txt"
    ds= parse_swahili(args.corpus_path)
    # ds= parse_pilecorpus(args.corpus_path)
    print("Length:", len(ds))
    # print("The sample of dataset is:", ds[:1000])
   
    seq_len = 256
    # top_k = 1000
    
    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model1)
    # tokenizer = GPT2Tokenizer.from_pretrained(args.model1)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    #add the SERENGETI model changes here:  
    # model1 = AutoModelForCausalLM.from_pretrained(args.model1, return_dict=True).to(device)
    # model1.config.pad_token_id = model1.config.eos_token_id
    # model2 = AutoModelForCausalLM.from_pretrained(args.model2, return_dict=True).to(device)
    # model2.eval() 
    #for pythia 
    model1 = GPTNeoXForCausalLM.from_pretrained(args.model1, return_dict=True).to(device)
    model1.config.pad_token_id = model1.config.eos_token_id
    model2 = GPTNeoXForCausalLM.from_pretrained(args.model2, return_dict=True).to(device)
    model2.eval()
    #for gpt-neo
    # model1 = GPTNeoForCausalLM.from_pretrained(args.model1, return_dict=True).to(device)
    # model1.config.pad_token_id = model1.config.eos_token_id
    # model2 = GPTNeoForCausalLM.from_pretrained(args.model2, return_dict=True).to(device)
    # model2.eval()

    samples = []
    prompts_list = []
    prompt_suffix=[]

    scores = {"mem":[], "XL": [], "S": [], "Lower": [], "zlib": []}

    num_batches = int(np.ceil(args.N / args.batch_size))
    
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
            #changed from 10 to 100
            input_len = 100
            input_ids = []
            attention_mask = []
            
            while len(input_ids) < args.batch_size:
                # Sample random text from the Pile corpus
                r = np.random.randint(0, len(ds))
                
                # print("*"*100)
                # print("The index Selected is:", r)
                
                
                prompt = " ".join(ds[r:r+100].split(" ")[1:-1])
                print("The length of a prompt is:", len(prompt))
                # print(prompt)
                prompt_suff=  " ".join(ds[r+100:r+200].split(" ")[1:-1])
                print("The length of the suffix is: ", len(prompt_suff))
                # print("The untruncated prompt is:",prompt)
                # print("The prompt suffix is: ", prompt_suff)
                # Tokenize the prompt ensuring consistent input lengths
                #removed padding="max_length" and max_length=input_len,
                inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len,  truncation=True)
                # print("the lenght of tokenized prompt is:", len(inputs))
                # print(inputs)
                prompt_suffix.append(prompt_suff)

                if len(inputs['input_ids'][0]) == input_len:
                    input_ids.append(inputs['input_ids'][0])
                    attention_mask.append(inputs['attention_mask'][0])
            # print("The input_ids are:", input_ids) 
            inputs = {'input_ids': torch.stack(input_ids), 
                      'attention_mask': torch.stack(attention_mask)}
            
            # The actual truncated prompts
            prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            print("The truncated prompt list is:", len(prompts))
            print(prompts)
            
            print("Attention Mask shape:", inputs['attention_mask'].shape)
        
            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                # top_k=top_k, 
                top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            prompts_list.append(prompts)
        
            # print("The prompt list is:", prompts_list[0][:2])
            # print("The prompt list is:", prompts_list)
            # print("The prompt suffix is:", prompt_suffix[0][:2])
            # print("len of prompts and suffix list:", len(prompts_list[0]), len(prompt_suffix))
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
    # print("*"*100)
    # print("Prompt List has the following prompts:",len(prompts_list[0]))
    scores["XL"] = np.asarray(scores["XL"])
    scores["S"] = np.asarray(scores["S"])
    scores["Lower"] = np.asarray(scores["Lower"])
    scores["zlib"] = np.asarray(scores["zlib"])

    model1_name = args.model1.replace("/", "_")
    model2_name = args.model2.replace("/", "_")

    
   
    
    sample_test = [s[:200] for s in samples]
    
    print("*"*1000)
    print("sample_test examples are:", sample_test)
    # print("the length of sample_test", len(sample_test))
    # print("*"*100)
    # print("prompt_suffix is :", prompt_suffix)
    # print("the length of sample_test", len(prompt_suffix))
    comparison_result = [1 if sample == prompt else 0 for sample, prompt in zip(sample_test, prompt_suffix)]
    # print("The comparison list length is:", len(comparison_result))
    ones_count = sum(comparison_result)
    total_count = len(comparison_result)
    memorization = (ones_count / total_count) * 100
    
    
    print("Memorization is: "  , memorization)
    prompts_list = [item for sublist in prompts_list for item in sublist]
    # print("*"*100)
    # print("Number of prompts are:", len(prompts_list))
    # # print("*"*100)
    # print("Number of Prompt Suffix are:", len(prompt_suffix))

    output_csv = f'output_scores_{model1_name}_{model2_name}.csv'
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['sample', 'prompt', 'suffix', 'memorized', 'PPL_XL', 'PPL_S', 'PPL_Lower', 'Zlib']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample,prompt,suff, mem, xl, s, lower, zlib_ in zip(samples, prompts_list, prompt_suffix, comparison_result, scores["XL"], scores["S"], scores["Lower"], scores["zlib"]):
            writer.writerow({'sample': sample, 'prompt': prompt, 'suffix': suff, 'memorized': mem, 'PPL_XL': xl, 'PPL_S': s, 'PPL_Lower': lower, 'Zlib': zlib_})

    print("Results saved to ", output_csv)
    
    output_txt = f'output_results_{model1_name}_{model2_name}.txt'
    with open(output_txt, 'w') as f:
        metric = -np.log(scores["XL"])
        f.write(f"======== top sample by XL perplexity: ========\n")
        f.write(print_best(metric, samples, "PPL", scores["XL"]))
        f.write("\n")

        metric = np.log(scores["S"]) / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of S and XL perplexities: ========\n")
        f.write(print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-S", scores["S"]))
        f.write("\n")

        metric = np.log(scores["Lower"]) / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of lower-case and normal-case perplexities: ========\n")
        f.write(print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["Lower"]))
        f.write("\n")

        metric = scores["zlib"] / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========\n")
        f.write(print_best(metric, samples, "PPL-XL", scores["XL"], "Zlib", scores["zlib"]))

        f.write(f"======== Percentage of memorization is: ========\n")
        f.write(f"========{memorization}")

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
