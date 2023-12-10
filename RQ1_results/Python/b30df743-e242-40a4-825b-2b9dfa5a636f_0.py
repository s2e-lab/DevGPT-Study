from github import Github

# Create a Github instance:
g = Github("<your-github-token>")

# Get the specific repo
repo = g.get_repo("<your-username>/<your-repo>")

def get_value(key):
    contents = repo.get_contents(key)
    return contents.decoded_content.decode()

def set_value(key, value):
    try:
        contents = repo.get_contents(key)
        repo.update_file(contents.path, "update", value, contents.sha)
    except:
        repo.create_file(key, "create", value)

# Usage
set_value("key1", "value1")
print(get_value("key1"))  # Outputs: value1
