# Your existing imports and database code here

# Initialize an empty list to hold the WCSS values
wcss = []

# Range of k values to try
k_range = range(1, 11)  # You can change this range based on your needs

# Loop through different k values to find the elbow
for k in k_range:
    clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=k, n_init="auto")
    
    # Your existing database fetch code
    rows = [
        (row[0], llm.decode(row[1]), row[2])
        for row in db.execute(
            """
        select id, embedding, content from embeddings
        where collection_id = (
            select id from collections where name = ?
        )
    """,
            [collection],
        ).fetchall()
    ]
    
    # Your existing code for fetching the embeddings
    to_cluster = np.array([item[1] for item in rows])
    
    # Fit the model
    clustering_model.fit(to_cluster)
    
    # Append the WCSS value for this k
    wcss.append(clustering_model.inertia_)

# Plotting the elbow graph
plt.figure()
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('The Elbow Method')
plt.show()
