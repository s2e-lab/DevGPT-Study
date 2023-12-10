import os
from transformers import BertTokenizer

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 512  # Adjust if using a different tokenizer with a different max length

def count_tokens_in_file(file_path):
    token_count = 0
    with open(file_path, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if line:  # to avoid empty lines
                # Tokenize and truncate if necessary
                data = tokenizer.encode(line, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
                token_count += len(data)

            # Print progress every 1000 lines
            if idx % 1000 == 0:
                print(f"Processed {idx} lines of {file_path}")
    return token_count

if __name__ == "__main__":
    directory = "datasets"
    files = [f for f in os.listdir(directory) if f.endswith(".jsonl")]
    print(f"Found {len(files)} JSONL files in the directory.")

    for filename in files:
        print(f"Processing {filename}...")
        file_path = os.path.join(directory, filename)
        tokens = count_tokens_in_file(file_path)
        print(f"Finished {filename}. Token Count: {tokens}")
    print("All files processed.")
