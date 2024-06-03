import torch
from datasets import load_dataset
import numpy as np
from pprint import pprint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_perplexity(sentence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10, output_file=None):
    idxs = np.argsort(metric)[::-1][:n]
    results = []

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            result = f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}"
        else:
            result = f"{i+1}: {name1}={scores1[idx]:.3f}, score={metric[idx]:.3f}"
        results.append(result)
        if output_file:
            output_file.write(result + '\n')
            output_file.write("\n")
            output_file.write(pprint(samples[idx]))
            output_file.write("\n\n")
        else:
            print(result)
            print()
            pprint(samples[idx])
            print()
            print()
    return results
