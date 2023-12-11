import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import hnswlib

def get_sentence_embeddings(sentences):
    # Load a pre-trained model
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    # Encode the sentences to get their embeddings
    embeddings = model.encode(sentences)

    return embeddings

def create_hnsw_index(embeddings, M=16, efC=100):
    # Create the HNSW index
    num_dim = embeddings.shape[1]
    index = hnswlib.Index(space='cosine', dim=num_dim)
    index.init_index(max_elements=embeddings.shape[0], ef_construction=efC, M=M)
    index.add_items(embeddings)

    return index

def load_hnsw_index(index_file):
    # Load the HNSW index from the specified file
    index = hnswlib.Index(space='cosine', dim=0)
    index.load_index(index_file)

    return index

def create_query_embedding(query, model):
    # Encode the query to get its embedding
    embedding = model.encode([query])[0]

    return embedding

def find_nearest_neighbors(index, query_embedding, k=5):
    # Find the k-nearest neighbors for the query embedding
    labels, distances = index.knn_query(query_embedding, k=k)

    return labels, distances

def rerank_chunks_with_cross_encoder(query, chunks, cross_encoder_model):
    # Create a list of tuples, each containing a query-chunk pair
    pairs = [(query, chunk) for chunk in chunks]

    # Get scores for each query-chunk pair using the cross encoder
    scores = cross_encoder_model.predict(pairs)

    # Sort the chunks based on their scores in descending order
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]

    return sorted_chunks

def perform_similarity_search(input_file, output_file, query=None, k=5):
    # Read the input Parquet file and create embeddings
    df = pd.read_parquet(input_file)
    embeddings = get_sentence_embeddings(df['chunk_content'].tolist())

    # Create the HNSW index and save it
    index = create_hnsw_index(embeddings)
    index.save_index(output_file)

    print("HNSW index created and saved successfully!")

    if query:
        # Load the HNSW index from the file
        loaded_index = load_hnsw_index(output_file)

        # Create an embedding for the query
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        query_embedding = create_query_embedding(query, model)

        # Find the k-nearest neighbors for the query
        labels, distances = find_nearest_neighbors(loaded_index, query_embedding, k=k)

        # Get the retrieved text chunks
        retrieved_chunks = [df['chunk_content'][label] for label in labels[0]]

        # Load the cross-encoder model
        cross_encoder_model = CrossEncoder('cross-encoder/model')

        # Re-rank the retrieved chunks with the cross-encoder model
        sorted_chunks = rerank_chunks_with_cross_encoder(query, retrieved_chunks, cross_encoder_model)

        return sorted_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to create and use an HNSW index for similarity search.')
    parser.add_argument('input_file', help='Input file containing text chunks in a Parquet format')
    parser.add_argument('output_file', help='Output file to save the HNSW index with .bin extension')
    parser.add_argument('--query', help='Query text for similarity search')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to retrieve')
    args = parser.parse_args()

    # Perform similarity search and get sorted_chunks
    sorted_chunks = perform_similarity_search(args.input_file, args.output_file, args.query, args.k)

    # Print the results
    if args.query and sorted_chunks:
        print(f"Query: {args.query}")
        for i, chunk in enumerate(sorted_chunks, start=1):
            print(f"Rank {i}:")
            print(f"Chunk Content: {chunk}")
            print()

