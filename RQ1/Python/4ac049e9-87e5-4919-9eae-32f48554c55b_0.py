def simple_generator():
    yield 1
    yield 2
    yield 3

for number in simple_generator():
    print(number)
