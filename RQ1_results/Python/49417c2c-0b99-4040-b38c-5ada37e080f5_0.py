import boto3
from botocore.client import Config

# Configure the endpoint URLs for different services
s3_endpoint_url = 'http://localhost:4572'
sqs_endpoint_url = 'http://localhost:4576'

# Create custom session objects with different endpoint URLs
s3_session = boto3.session.Session()
sqs_session = boto3.session.Session()

# Register event listeners for S3 to set the endpoint URL
s3_session.client('s3').meta.events.register(
    'service-created.s3',
    lambda event, **kwargs: event.add_to_service(
        'endpoint_url', s3_endpoint_url
    )
)

# Register event listeners for SQS to set the endpoint URL
sqs_session.client('sqs').meta.events.register(
    'service-created.sqs',
    lambda event, **kwargs: event.add_to_service(
        'endpoint_url', sqs_endpoint_url
    )
)

# Set the custom sessions as default session factories
boto3.setup_default_session(
    region_name='us-east-1', 
    botocore_session=s3_session,
    session=boto3.DEFAULT_SESSION
)

boto3.setup_default_session(
    region_name='us-east-1', 
    botocore_session=sqs_session,
    session=boto3.DEFAULT_SESSION
)

# Now all subsequent client/resource creation will use the registered sessions
s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')

# Use the S3 client and SQS client with the custom endpoint URLs
s3_client.list_buckets()
sqs_client.list_queues()
