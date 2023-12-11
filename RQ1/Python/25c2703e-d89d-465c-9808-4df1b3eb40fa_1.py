import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util

# Initialize the transformer model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Load the embeddings from the numpy file
embeddings = np.load('words_emb.npy')

# Load the original data to access words
with open("words.pkl", "rb") as file:
    data = pickle.load(file)

# Define the search keyword
search_keyword = "example"  # change this to your search keyword

# Find the top 100 most similar word vectors
query_embedding = model.encode(search_keyword)
cos_similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
top_results = np.argpartition(-cos_similarities, range(100))[:100]

print("Top 100 most similar word entries to '{}':".format(search_keyword))
for idx in top_results:
    print("Word: {}, Cosine Similarity: {:.4f}".format(data[idx]['word'], cos_similarities[idx]))
