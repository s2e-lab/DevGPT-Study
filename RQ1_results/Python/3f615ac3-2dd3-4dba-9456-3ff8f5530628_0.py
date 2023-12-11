from sqlalchemy import create_engine
import pandas as pd

# Define database parameters
username = 'your_username'
password = 'your_password'
host = 'your_host'
database = 'your_database'

# Create a SQL Alchemy engine
engine = create_engine(f'postgresql://{username}:{password}@{host}/{database}')

# Execute SQL query and convert it into a DataFrame
df = pd.read_sql("SELECT stationid, test_stationid FROM lu_teststation", engine)

# Close the connection
engine.dispose()

# Convert DataFrame to dictionary
test_station_renaming_key = pd.Series(df.test_stationid.values,index=df.stationid).to_dict()

print(test_station_renaming_key)
