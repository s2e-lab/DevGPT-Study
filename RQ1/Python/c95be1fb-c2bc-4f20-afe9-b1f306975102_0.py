import numpy as np
import altair as alt
import pandas as pd

# Generate data points
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))

# Create a DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Create an Altair chart
chart = alt.Chart(df).mark_line().encode(
    x='x',
    y='y'
).properties(
    width=400,
    height=300
)

# Display the chart
chart.interactive()
