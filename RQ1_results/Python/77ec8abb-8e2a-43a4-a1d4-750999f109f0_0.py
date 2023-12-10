import matplotlib.pyplot as plt
import numpy as np

# Assuming 'corelation' is the correlation matrix (NumPy array) obtained from pandas DataFrame

# Get the column names for x and y tick labels
columns = corelation.columns

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the heatmap
heatmap = ax.imshow(corelation, cmap='coolwarm', interpolation='nearest')

# Add the colorbar
cbar = plt.colorbar(heatmap)

# Set the tick positions and labels for both x and y axes
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks(np.arange(len(columns)))
ax.set_xticklabels(columns)
ax.set_yticklabels(columns)

# Rotate the tick labels on the x-axis for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add annotations (correlation values) to the heatmap cells
for i in range(len(columns)):
    for j in range(len(columns)):
        text = ax.text(j, i, f"{corelation.iloc[i, j]:.2f}", ha="center", va="center", color="w")

# Set axis labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Features")
plt.title("Correlation Heatmap")

# Show the plot
plt.show()
