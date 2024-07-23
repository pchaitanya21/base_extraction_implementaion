from transformers import BertForMaskedLM, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, DistilBertForMaskedLM, DistilBertTokenizer, RobertaForMaskedLM, RobertaTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import math
import operator
from heapq import nlargest
import argparse
import pickle
import time
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--proc-id', type=int)
parser.add_argument('--model', type=str, choices=['bert', 'distilbert', 'roberta'])
parser.add_argument('--dataset', type=str, choices=['swa', 'fin', 'eng'])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_pilecorpus(path):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    # with open(wet_file) as f:
        # lines = f.readlines() 
    
    # start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
    all_texts = ""
    dataset = load_dataset(path, split="train", streaming=True)
    shuffled_dataset = dataset.shuffle(seed=42)
    #len(dataset['train'])
    dataset_head= shuffled_dataset.skip(0)
    dataset_head = shuffled_dataset.take(1000000)
    for text in dataset_head:
        all_texts+= text['text']
    # for i in range(10):
      # all_texts+= dataset['train']['translation'][i]['bg']
      # print("done")
      # all_texts+= dataset['train']['translation'][i]['cs']
    # count_eng = 0
    # for i in range(len(start_idxs)-1):
    #     start = start_idxs[i]
    #     end = start_idxs[i+1]
    #     if "WARC-Identified-Content-Language: eng" in lines[start+7]:
    #         count_eng += 1
    #         for j in range(start+10, end):
    #             all_eng += lines[j]

    return all_texts

def parse_lang(path):
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

attack_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
attack_tokenizer.pad_token = attack_tokenizer.eos_token

if args.dataset == 'eng':
    attack_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
elif args.dataset == 'fin':
    attack_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
elif args.dataset == 'swa':
    attack_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

attack_model = attack_model.to('cuda:0')

if args.model == 'bert':
    search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    search_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

elif args.model == 'distilbert':
    search_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    search_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

elif args.model == 'roberta':
    search_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    search_model = RobertaForMaskedLM.from_pretrained('roberta-base')

print(search_model)

search_model = search_model.to('cuda:1')

token_dropout = torch.nn.Dropout(p=0.7)

if args.dataset == 'eng':
    texts = parse_pilecorpus('monology/pile-uncopyrighted')
elif args.dataset == 'fin':
    texts = parse_lang('fin_sample.txt')
elif args.dataset == 'swa':
    texts = parse_lang('swa_sample.txt')



def generate_neighbours_alt(tokenized, num_word_changes=1):
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        if args.model == 'bert':
            embeds = search_model.bert.embeddings(text_tokenized)
        elif args.model == 'distilbert':
            embeds = search_model.distilbert.embeddings(text_tokenized)
        elif args.model == 'roberta':
            embeds = search_model.roberta.embeddings(text_tokenized)
            
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 6, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:

                #alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda:1'), text_tokenized[:,target_token_index+1:]), dim=1)
                #alt_text = search_tokenizer.batch_decode(alt)[0]
                if original_prob.item() == 1:
                    print("probability is one!")
                    replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                else:
                    replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())

    
    #highest_scored_texts = max(candidate_scores.iteritems(), key=operator.itemgetter(1))[:100]
    highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)

    replacement_keys = nlargest(50, replacements, key=replacements.get)
    replacements_new = dict()
    for rk in replacement_keys:
        replacements_new[rk] = replacements[rk]
    
    replacements = replacements_new
    print("got highest scored single texts, will now collect doubles")

    highest_scored = nlargest(100, replacements, key=replacements.get)


    texts = []
    for single in highest_scored:
        alt = text_tokenized
        target_token_index, cand = single
        alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda:1'), alt[:,target_token_index+1:]), dim=1)
        alt_text = search_tokenizer.batch_decode(alt)[0]
        texts.append((alt_text, replacements[single]))


    return texts




def generate_neighbours(tokenized, num_word_changes=1):
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        embeds = search_model.bert.embeddings(text_tokenized)
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 10, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:

                alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda:1'), text_tokenized[:,target_token_index+1:]), dim=1)
                alt_text = search_tokenizer.batch_decode(alt)[0]
                candidate_scores[alt_text] = prob/(1-original_prob)
                replacements[(target_token_index, cand)] = prob/(1-original_prob)

    
    highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)



    return highest_scored_texts



def get_logprob(text):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:0')
    logprob = - attack_model(text_tokenized, labels=text_tokenized).loss.item()

    return logprob




def get_logprob_batch(text):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:0')

    ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=attack_tokenizer.pad_token_id)
    logits = attack_model(text_tokenized, labels=text_tokenized).logits[:,:-1,:].transpose(1,2)
    manual_logprob = - ce_loss(logits, text_tokenized[:,1:])
    mask = manual_logprob!=0
    manual_logprob_means = (manual_logprob*mask).sum(dim=1)/mask.sum(dim=1)


    return manual_logprob_means.tolist()


all_scores = []

if args.dataset == 'eng':
    batch_size = 3000
elif args.dataset == 'fin':
    batch_size = 1200
elif args.dataset == 'swa':
    batch_size = 1200
for text in tqdm(texts[args.proc_id*batch_size:(args.proc_id+1)*batch_size]):
    attack_model.eval()
    search_model.eval()

    tok_orig = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")


    scores = dict()
    scores[f'<original_text>: {orig_dec}'] = get_logprob(orig_dec)

    with torch.no_grad():
        start = time.time()
        #neighbours = generate_neighbours(text)
        neighbours = generate_neighbours_alt(text)
        end = time.time()
        print("generating neighbours took seconds:", end-start)

        # one_word_neighbours
        for i, neighbours in enumerate([neighbours]):
            neighbours_texts = []
            for n in neighbours:
                neighbours_texts.append((n[0].replace(" [SEP]", " ").replace("[CLS] ", " "), n[1]))
                score = get_logprob_batch([n[0].replace(" [SEP]", " ").replace("[CLS] ", " ")])
                scores[n] = score


            if i == 0:
                scores_temp = scores        
    
    all_scores.append(scores_temp)

with open(f'all_scores_{args.dataset}_{args.model}_{args.proc_id}.pkl', 'wb') as file:
    pickle.dump(all_scores, file)


all_scores = []

for text in tqdm(texts[args.proc_id*1200:(args.proc_id+1)*1200]):
    attack_model.eval()
    search_model.eval()

    tok_orig = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")


    scores = dict()
    scores[f'<original_text>: {orig_dec}'] = get_logprob(orig_dec)

    with torch.no_grad():
        neighbours = generate_neighbours(text)

        for n in neighbours:
            n = n.replace(" [SEP]", " ").replace("[CLS] ", " ")
            scores[n] = get_logprob(n)

    all_scores.append(scores)



with open(f'all_scores{args.proc_id}.pkl', 'wb') as file:
    pickle.dump(all_scores, file)
