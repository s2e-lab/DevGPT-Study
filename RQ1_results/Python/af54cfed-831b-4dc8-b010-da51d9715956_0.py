# Import required libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load the data (you can replace this part with the code to load your melted and grouped data)
# ... (your data loading and melting code here)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(id='map'),
    dcc.Dropdown(
        id='gwp-dropdown',
        options=[{'label': i, 'value': i} for i in melted_df['gwp'].unique()],
        value=20  # initial value
    ),
    dcc.Dropdown(
        id='metric-dropdown',
        options=[{'label': i, 'value': i} for i in melted_df['Emissions Metric'].unique()],
        value='CH4'  # initial value
    ),
    dcc.Graph(id='bar-chart')
])

# Define callback to update both map and bar chart
@app.callback(
    [Output('map', 'figure'),
     Output('bar-chart', 'figure')],
    [Input('map', 'clickData'),
     Input('gwp-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_figures(clickData, selected_gwp, selected_metric):
    # Filter data based on selected GWP and metric
    filtered_df = melted_df[(melted_df['gwp'] == selected_gwp) & (melted_df['Emissions Metric'] == selected_metric)]
    grouped_filtered_df = filtered_df.groupby(['Field Name', 'Latitude', 'Longitude', 'Year', 'gwp']).sum().reset_index()

    # Create and return the new map figure
    fig_map = px.scatter_mapbox(
        grouped_filtered_df,
        lat="Latitude",
        lon="Longitude",
        size="Emissions Intensity Value",
        color="Emissions Intensity Value",
        hover_name="Field Name",
        hover_data=["Year", "gwp", "Emissions Intensity Value"],
        animation_frame="Year",
        color_continuous_scale=px.colors.sequential.Plasma,
        size_max=20,
        zoom=1,
        title="Emissions Intensity by Field"
    )
    fig_map.update_layout(mapbox_style="open-street-map")

    # If a field is selected on the map
    if clickData:
        field_name = clickData['points'][0]['hovertext']
        filtered_df = filtered_df[filtered_df['Field Name'] == field_name]

    # Create and return the new bar chart figure
    fig_bar = px.bar(
        filtered_df,
        x='Year',
        y='Emissions Intensity Value',
        color='Supply Chain Segment'
    )

    return fig_map, fig_bar

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
