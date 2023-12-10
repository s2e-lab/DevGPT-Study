# Using frozenset as a key in a dictionary
my_set = {1, 2, 3}
my_dict = {frozenset(my_set): "value"}

print(my_dict)  # Output: {frozenset({1, 2, 3}): 'value'}
