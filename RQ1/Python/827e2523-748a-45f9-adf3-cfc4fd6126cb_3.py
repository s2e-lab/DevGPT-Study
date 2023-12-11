from collections import defaultdict

# Creating a defaultdict with default value as 0
d = defaultdict(int)

d['a'] += 1
d['b'] += 2
print(d['c'])  # Output: 0 (since 'c' does not exist, it returns the default value 0)
