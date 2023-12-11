code = """
a = 1
b = 2
result = a + b
"""

locals_ = {}
exec(code, {}, locals_)
print(locals_['result'])  # Output: 3
