import logging

logging.basicConfig(
    filename="media_cli.log",
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

try:
    # Code that may raise an error
except Exception as e:
    logging.error(f"An error occurred: {e}")
