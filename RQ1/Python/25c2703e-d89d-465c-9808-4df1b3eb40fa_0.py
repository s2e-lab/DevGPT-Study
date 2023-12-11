import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Initialize the transformer model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Load data from the pkl file
with open("words.pkl", "rb") as file:
    data = pickle.load(file)

# Placeholder for embeddings
embeddings = []

for entry in data:
    # Get embeddings for the keywords of each word
    keyword_embeddings = model.encode(entry['keywords'])
    
    # Compute the spherical mean
    norm_keyword_embeddings = normalize(keyword_embeddings)
    mean_embedding = norm_keyword_embeddings.mean(axis=0)
    mean_embedding /= np.linalg.norm(mean_embedding)
    
    # Add the mean embedding to the list
    embeddings.append(mean_embedding)

# Convert list of embeddings to numpy array and change dtype to float16
embeddings = np.array(embeddings, dtype=np.float16)

# Save the embeddings to a numpy file
np.save('words_emb.npy', embeddings)
