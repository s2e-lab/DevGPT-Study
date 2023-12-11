import os

def main():
    environment = os.environ.get("MY_APP_ENV", "prod")
    config = EnvironmentConfig(environment)
    query_instance = SampleTablesQuery(config)
    # Rest of the code

if __name__ == "__main__":
    main()
