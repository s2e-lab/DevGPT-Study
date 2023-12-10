import pprint

my_list = [1, 2, 3, [4, 5, 6], {'a': 7, 'b': 8, 'c': 9}]

# Create a PrettyPrinter object with the float parameter set to 'float'
pp = pprint.PrettyPrinter(indent=4, float='float')

# Pretty print the list
pp.pprint(my_list)
