import os
from transformers import BertTokenizer

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def count_tokens_in_file(file_path):
    token_count = 0
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # to avoid empty lines
                data = tokenizer.encode(line, add_special_tokens=False)
                token_count += len(data)
    return token_count

if __name__ == "__main__":
    directory = "datasets"
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            tokens = count_tokens_in_file(file_path)
            print(f"Filename: {filename}, Token Count: {tokens}")
