import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def copy_file(source, destination):
    shutil.copy2(source, destination)

def move_file(source, destination):
    shutil.move(source, destination)

def delete_file(path):
    os.remove(path)

def trash_file(path):
    # Implement your own logic to move the file to trash

if __name__ == '__main__':
    # Input data
    file_operations = [
        ('copy', 'source_file_1', 'destination_1'),
        ('move', 'source_file_2', 'destination_2'),
        ('delete', 'file_to_delete'),
        ('trash', 'file_to_trash')
    ]

    # Create a thread pool executor
    with ThreadPoolExecutor() as executor:
        for operation, *args in file_operations:
            if operation == 'copy':
                executor.submit(copy_file, *args)
            elif operation == 'move':
                executor.submit(move_file, *args)
            elif operation == 'delete':
                executor.submit(delete_file, *args)
            elif operation == 'trash':
                executor.submit(trash_file, *args)
