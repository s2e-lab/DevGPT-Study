def save_to_file(filename, data):
    with open(filename, 'a') as file:
        file.write(data)
