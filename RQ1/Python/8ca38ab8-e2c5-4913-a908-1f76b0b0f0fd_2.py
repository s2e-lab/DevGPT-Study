string_with_null = "Hello\x00World"

# Write the string to a file
with open("data.txt", "w") as file:
    file.write(string_with_null)

# Read the string from the file
with open("data.txt", "r") as file:
    string_from_file = file.read()
