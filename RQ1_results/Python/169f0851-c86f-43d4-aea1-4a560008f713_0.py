import boto3

def get_caller_id():
    # Create a Boto3 client for the AWS Security Token Service (STS)
    sts_client = boto3.client('sts')

    # Call the GetCallerIdentity API to retrieve information about the AWS account
    response = sts_client.get_caller_identity()

    # Extract and return the AWS Account ID from the response
    account_id = response['Account']
    return account_id
