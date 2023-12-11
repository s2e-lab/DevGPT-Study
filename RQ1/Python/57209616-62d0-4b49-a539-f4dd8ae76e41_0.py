def print_length(lst):
    def new_len(x):
        print(len(x))
    def new_print(x):
        return len(x)
    new_len(lst)
