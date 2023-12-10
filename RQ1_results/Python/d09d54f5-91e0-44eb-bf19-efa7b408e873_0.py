from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Normalize latitude and longitude
coords = data[['f__Latitude', 'f__Longitude']]
coords_scaled = StandardScaler().fit_transform(coords)

# Define DBSCAN
db = DBSCAN(eps=0.3, min_samples=10)

# Fit and predict clusters
data['Cluster'] = db.fit_predict(coords_scaled)
