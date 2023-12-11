from collections import Counter

# Counting occurrences of elements in a list
my_list = [1, 2, 3, 1, 2, 1, 4, 5, 4]
counted_items = Counter(my_list)

print(counted_items)  # Output: Counter({1: 3, 2: 2, 4: 2, 3: 1, 5: 1})
