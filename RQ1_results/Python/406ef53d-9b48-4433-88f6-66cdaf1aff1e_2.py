def get_post():
    response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
    post = response.json()
    return post
