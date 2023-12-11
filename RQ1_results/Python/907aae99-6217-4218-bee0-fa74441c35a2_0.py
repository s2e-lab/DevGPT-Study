import numpy as np

# Your large numpy array
arr = np.arange(1000)

# Convert to string with custom edgeitems
arr_str = np.array2string(arr, edgeitems=10)

print(arr_str)
