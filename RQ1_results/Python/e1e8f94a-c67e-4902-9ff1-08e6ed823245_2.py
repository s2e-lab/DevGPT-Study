def dfs(node):
    print(node.value)  # Process the node value
    for child in node.children:
        dfs(child)

dfs(root)
