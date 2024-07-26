import re

def tokenize(text):
    """
    Tokenizes the input text into words.
    """
    # Simple word tokenization (can be replaced with a more advanced tokenizer if needed)
    return re.findall(r'\b\w+\b', text)

def process_file(input_file, output_file, min_tokens=20):
    """
    Reads the input file, deduplicates sentences longer than min_tokens,
    and writes the result to the output file.
    """
    seen_sentences = set()
    deduplicated_sentences = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            sentences = re.split(r'(?<=[.!?]) +', line.strip())
            for sentence in sentences:
                tokens = tokenize(sentence)
                if len(tokens) > min_tokens:
                    sentence_normalized = ' '.join(tokens)
                    if sentence_normalized not in seen_sentences:
                        seen_sentences.add(sentence_normalized)
                        deduplicated_sentences.append(sentence)

    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in deduplicated_sentences:
            file.write(sentence + '\n')

# Example usage
input_file = '/content/drive/MyDrive/fin_sample.txt'  # Path to your input file
output_file = 'fin_dedup_20.txt'  # Path to your output file
process_file(input_file, output_file)