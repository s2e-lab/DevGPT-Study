from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Use the environment variables
database_url = os.getenv('DATABASE_URL')
secret_key = os.getenv('SECRET_KEY')
debug = os.getenv('DEBUG')

print(database_url, secret_key, debug)
