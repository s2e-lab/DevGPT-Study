# Create the initial map
fig_map = px.scatter_mapbox(
    grouped_df,
    lat="Latitude",
    lon="Longitude",
    # ... (as before)
)

# Create the initial bar chart with total emissions
fig_bar = px.bar(
    melted_df,
    x='Year',
    y='Emissions Intensity Value',
    color='Supply Chain Segment',
    animation_frame='Year'
)
