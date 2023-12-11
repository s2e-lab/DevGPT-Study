import boto3
import random
import string

def create_iam_user(username):
    iam_client = boto3.client('iam')

    # Generate a random password
    password = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(12))

    # Create the IAM user
    response = iam_client.create_user(UserName=username)

    # Attach the AdministratorAccess policy to the user
    iam_client.attach_user_policy(
        UserName=username,
        PolicyArn='arn:aws:iam::aws:policy/AdministratorAccess'
    )

    # Create access key for the user
    access_key_response = iam_client.create_access_key(UserName=username)
    access_key = access_key_response['AccessKey']
    
    # Update the password for the user
    iam_client.create_login_profile(
        UserName=username,
        Password=password,
        PasswordResetRequired=False
    )
    
    # Return the user details
    return {
        'UserName': username,
        'Password': password,
        'AccessKeyId': access_key['AccessKeyId'],
        'SecretAccessKey': access_key['SecretAccessKey']
    }

# Provide the desired username for the new IAM user
new_username = 'example_user'

# Create the IAM user with the AdministratorAccess policy attached and generate access key
user_details = create_iam_user(new_username)

# Print the user details
print("User created successfully.")
print(f"Username: {user_details['UserName']}")
print(f"Password: {user_details['Password']}")
print(f"Access Key ID: {user_details['AccessKeyId']}")
print(f"Secret Access Key: {user_details['SecretAccessKey']}")
