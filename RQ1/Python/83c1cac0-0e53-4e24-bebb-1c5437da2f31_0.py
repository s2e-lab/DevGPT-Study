def magic_function(iterable):
    it = iter(iterable)
    try:
        last = next(it)
    except StopIteration:
        return
    while True:
        try:
            current = next(it)
            yield last, False
            last = current
        except StopIteration:
            yield last, True
            break
