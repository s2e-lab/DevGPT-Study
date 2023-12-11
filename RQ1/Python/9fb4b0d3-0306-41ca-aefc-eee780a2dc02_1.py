matrices = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])]
cutoff = 0.5

print(max_nonzero_per_row(matrices, cutoff))  # Outputs: [2, 1]
