def convert_to_quoted(s):
    # Check if the string spans multiple lines
    if "\n" in s:
        # Escape triple double quotes
        s = s.replace('"""', '\\"\\"\\"')
        # Now wrap the whole string into triple double quotes
        return f'"""{s}"""'
    else:
        # Escape double quotes
        s = s.replace('"', '\\"')
        # Now wrap the whole string into double quotes
        return f'"{s}"'
