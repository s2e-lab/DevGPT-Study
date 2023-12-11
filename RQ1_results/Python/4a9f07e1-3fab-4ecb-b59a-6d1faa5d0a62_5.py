# Creating a base map
map = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=2)

# Creating a marker cluster
marker_cluster = MarkerCluster().add_to(map)

# Adding markers to the cluster
for i, row in data.iterrows():
    folium.Marker(location=[row['latitude'], row['longitude']],
                  popup=row['Orte']).add_to(marker_cluster)

# Display the map
map
