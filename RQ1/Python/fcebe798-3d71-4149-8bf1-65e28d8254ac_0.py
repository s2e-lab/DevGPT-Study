class MyClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

obj = MyClass(10, 20)
print(obj.__dict__)  # Outputs: {'x': 10, 'y': 20}
