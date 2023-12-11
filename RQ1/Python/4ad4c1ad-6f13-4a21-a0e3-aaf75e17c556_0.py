import copy

original_list = [[1, 2, 3], [4, 5, 6]]
deep_copied_list = copy.deepcopy(original_list)

# Now, if you modify the nested list inside the original list...
original_list[0][0] = 99

print(original_list)        # Outputs: [[99, 2, 3], [4, 5, 6]]
print(deep_copied_list)     # Outputs: [[1, 2, 3], [4, 5, 6]] (remains unchanged)
