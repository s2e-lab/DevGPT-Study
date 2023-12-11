import boto3
import time

def get_roles_with_policies(policy_arns):
    client = boto3.client('iam')
    paginator = client.get_paginator('list_roles')
    roles_with_policies = []

    for page in paginator.paginate():
        for role in page['Roles']:
            role_policies = get_role_policies(client, role['RoleName'])
            if set(policy_arns).issubset(set(role_policies)):
                roles_with_policies.append(role['RoleName'])

    return roles_with_policies

def get_role_policies(client, role_name):
    policy_arns = []
    try:
        attached_policies = client.list_attached_role_policies(RoleName=role_name)['AttachedPolicies']
        for policy in attached_policies:
            policy_arns.append(policy['PolicyArn'])
    except Exception as e:
        print(f"Failed to fetch policies for role: {role_name} due to: {str(e)}")
        time.sleep(1)  # basic backoff strategy
    return policy_arns

# list of policy ARNs we want to check
policy_arns = ['arn:aws:iam::aws:policy/AmazonS3FullAccess', 'arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess']

# get roles with the specified policies
roles = get_roles_with_policies(policy_arns)

# print the roles
for role in roles:
    print(role)
