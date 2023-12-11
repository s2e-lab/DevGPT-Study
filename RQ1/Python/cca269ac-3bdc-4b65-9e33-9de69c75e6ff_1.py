from utils import get_file_or_directory_path

@click.command()
def some_command():
    """An example command that uses get_file_or_directory_path."""
    file_path = get_file_or_directory_path("Enter a file path: ")
    directory_path = get_file_or_directory_path("Enter a directory path: ", is_directory=True)

    click.echo(f"File path: {file_path}")
    click.echo(f"Directory path: {directory_path}")

if __name__ == '__main__':
    some_command()
