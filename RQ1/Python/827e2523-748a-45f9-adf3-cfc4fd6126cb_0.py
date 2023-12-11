from collections import namedtuple

# Creating a namedtuple called 'Point' with fields 'x' and 'y'
Point = namedtuple('Point', ['x', 'y'])

# Creating an instance of the Point namedtuple
p = Point(x=1, y=2)

print(p.x)  # Output: 1
print(p.y)  # Output: 2
