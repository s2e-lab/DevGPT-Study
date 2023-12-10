import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.getenv('API_KEY')
db_password = os.getenv('DB_PASSWORD')
debug = os.getenv('DEBUG')

# Example usage
print(f"API Key: {api_key}")
print(f"Database Password: {db_password}")
print(f"Debug Mode: {debug}")
