import copy

original_list = [[1, 2, 3], [4, 5, 6]]
shallow_copy_list = copy.copy(original_list)
deep_copy_list = copy.deepcopy(original_list)

original_list[0][0] = 99

print(original_list)     # Output: [[99, 2, 3], [4, 5, 6]]
print(shallow_copy_list)  # Output: [[99, 2, 3], [4, 5, 6]]
print(deep_copy_list)     # Output: [[1, 2, 3], [4, 5, 6]]
