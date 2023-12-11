from collections import deque

# Creating a deque
d = deque([1, 2, 3])

d.append(4)       # Add an element to the right end
d.appendleft(0)   # Add an element to the left end

print(d)  # Output: deque([0, 1, 2, 3, 4])
