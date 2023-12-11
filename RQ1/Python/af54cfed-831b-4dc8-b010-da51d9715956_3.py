@app.callback(
    Output('bar-chart', 'figure'),
    [Input('map', 'clickData'),
     Input('gwp-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_bar_chart(clickData, selected_gwp, selected_metric):
    # Filter data based on selected GWP and metric
    filtered_df = melted_df[(melted_df['gwp'] == selected_gwp) & (melted_df['Emissions Metric'] == selected_metric)]
    
    # If a field is selected on the map
    if clickData:
        field_name = clickData['points'][0]['hovertext']
        filtered_df = filtered_df[filtered_df['Field Name'] == field_name]
        
    # Create and return the new bar chart figure
    fig_bar = px.bar(
        filtered_df,
        x='Year',
        y='Emissions Intensity Value',
        color='Supply Chain Segment',
        animation_frame='Year'
    )
    return fig_bar
