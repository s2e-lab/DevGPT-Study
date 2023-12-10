import click
import os

def get_file_or_directory_path(message, is_directory=False):
    """Get a file or directory path from user input."""
    while True:
        path = input(message)
        if not os.path.exists(path):
            click.echo("Path does not exist. Please enter a valid path.")
            continue

        if is_directory and not os.path.isdir(path):
            click.echo("Path is not a directory. Please enter a directory path.")
            continue

        if not is_directory and not os.path.isfile(path):
            click.echo("Path is not a file. Please enter a file path.")
            continue

        return path
