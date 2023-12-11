try:
    # Attempt to perform an operation
except SomeCustomError as e:
    click.echo(f"An error occurred: {e}")
    click.echo("Possible solution: ...")
