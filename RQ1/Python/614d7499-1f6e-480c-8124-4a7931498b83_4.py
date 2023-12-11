import json

def main():
    with open('config.json', 'r') as f:
        config_data = json.load(f)
        
    environment = config_data.get('environment', 'prod')
    config = EnvironmentConfig(environment)
    query_instance = SampleTablesQuery(config)
    # Rest of the code

if __name__ == "__main__":
    main()
