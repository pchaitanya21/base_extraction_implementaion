import re

def deduplicate_text(file_path, output_path, min_length=50):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Dictionary to track positions and counts of substrings
    substr_counts = {}
    deduplicated_text = text
    substr_length = min_length

    # Sliding window to extract all substrings of length min_length
    for i in range(len(text) - substr_length + 1):
        substr = text[i:i + substr_length]
        if substr in substr_counts:
            substr_counts[substr].append(i)
        else:
            substr_counts[substr] = [i]

    # Create a list of positions to be removed
    remove_positions = []
    for substr, positions in substr_counts.items():
        if len(positions) > 1:
            remove_positions.extend(positions[1:])
    
    # Sort positions in reverse order to avoid shifting issues
    remove_positions.sort(reverse=True)
    
    # Remove repeated substrings
    for pos in remove_positions:
        deduplicated_text = deduplicated_text[:pos] + deduplicated_text[pos + substr_length:]
    
    # Write the cleaned text to a new file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(deduplicated_text)

# Define file paths
input_file = 'swa_sample.txt'
output_file = 'swa_dedup.txt'

# Deduplicate the text
deduplicate_text(input_file, output_file)
