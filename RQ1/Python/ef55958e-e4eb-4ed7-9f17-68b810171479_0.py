# Tokenization, Building Vocabulary and Encoding
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(dataset.data)

# Convert the sparse matrix to dense and then to PyTorch tensors
vectors = vectors.todense()
vectors_tensor = torch.from_numpy(vectors)

# Convert targets to PyTorch tensors
targets_tensor = torch.tensor(dataset.target)
