import torchtext.datasets as datasets

imdb_dataset = datasets.IMDB()
train_dataset = imdb_dataset.train
test_dataset = imdb_dataset.test

# Accessing data from the dataset
train_examples = train_dataset.examples
test_examples = test_dataset.examples

# Example usage of the data
for example in train_examples[:5]:
    text = example.text
    label = example.label
    print(f"Text: {text}\nLabel: {label}\n")
