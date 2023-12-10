# Extracting latitude and longitude columns
X = data[['latitude', 'longitude']]

# Creating KMeans model
kmeans = KMeans(n_clusters=5)  # You can change the number of clusters
data['cluster'] = kmeans.fit_predict(X)
