app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='map', figure=fig_map),
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
