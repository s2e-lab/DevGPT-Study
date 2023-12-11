import geopandas as gpd
import folium

# Read the shapefile
shapefile_path = 'path_to_your_shapefile.shp'
gdf = gpd.read_file(shapefile_path)

# Create a Folium map object
m = folium.Map()

# Convert the GeoDataFrame to GeoJSON format
geojson_data = gdf.to_json()

# Add the GeoJSON data to the map as a GeoJson layer
folium.GeoJson(geojson_data).add_to(m)

# Save the map as an HTML file or display it inline
m.save('map.html')
# OR
m
