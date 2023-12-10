import matplotlib.pyplot as plt
import numpy as np

# Assume 'data' is your 2D matrix
data = np.random.rand(80, 80)

# Size of the matrix
height, width = data.shape

# Create a figure
fig, ax = plt.subplots()

# Display the image, setting extent to match data shape
im = ax.imshow(data, aspect='equal', interpolation='none',
               extent=[0, width, 0, height])

# Add labels, title, etc. here if you like
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Matrix Visualization")

# Adjust figure size to match the axes
fig.canvas.draw()  # needed to get correct sizes before saving
xlim = ax.get_xlim()
ylim = ax.get_ylim()
fig.set_size_inches((xlim[1]-xlim[0])/80, (ylim[1]-ylim[0])/80, forward=True)

# Save the figure
fig.savefig('output_exact.png', dpi=80)

plt.close(fig)
