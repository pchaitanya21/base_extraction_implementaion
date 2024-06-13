import torch
from datasets import load_dataset
import numpy as np
from pprint import pprint, pformat


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_perplexity(sentence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    Print the `n` best samples according to the given `metric`.
    Returns a string containing the information for each sample.
    """
    idxs = np.argsort(metric)[::-1][:n]
    output_string = ""

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            sample_info = f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}"
        else:
            sample_info = f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}"

        sample_text = samples[idx]
        output_string += sample_info + "\n" + sample_text + "\n\n"

    return output_string