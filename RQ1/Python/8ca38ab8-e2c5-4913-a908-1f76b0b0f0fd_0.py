string_with_null = "Hello\x00World"
string_without_null = string_with_null.replace('\x00', '')  # Remove null character
print(string_without_null)  # Output: HelloWorld
