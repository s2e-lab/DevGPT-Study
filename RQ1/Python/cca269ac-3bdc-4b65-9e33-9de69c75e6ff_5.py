try:
    # Code that may raise an error
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
    click.echo("An unexpected error occurred. Please report this issue.")
