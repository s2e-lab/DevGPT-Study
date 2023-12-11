# Python 2
u_string = u"abc"  # Unicode string
U_string = U"abc"  # Raw Unicode string

print(u_string)  # Output: abc
print(U_string)  # Output: abc

# Backslash is treated as an escape character
u_string = u"\n"  # Unicode string
U_string = U"\n"  # Raw Unicode string

print(u_string)  # Output: (newline character)
print(U_string)  # Output: \n (literal backslash followed by 'n')
