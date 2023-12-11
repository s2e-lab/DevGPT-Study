import torchtext.datasets as datasets

# Define the root directory where the dataset will be stored
root = '.data'

# Load the IMDB dataset
train_dataset, test_dataset = datasets.IMDB(root=root, split=('train', 'test'))

# Accessing the train dataset
train_examples = train_dataset.examples
train_fields = train_dataset.fields

# Accessing the test dataset
test_examples = test_dataset.examples
test_fields = test_dataset.fields
