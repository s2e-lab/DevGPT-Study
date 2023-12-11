import argparse

def main():
    parser = argparse.ArgumentParser(description="Set up environment.")
    parser.add_argument('--env', default='prod', help='Environment: dev or prod')
    args = parser.parse_args()

    config = EnvironmentConfig(args.env)
    query_instance = SampleTablesQuery(config)
    # Rest of the code

if __name__ == "__main__":
    main()
