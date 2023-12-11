# Fetching the tree

def get_tree(node, depth=0):
    indent = '    ' * depth
    print(f"{indent}{node.name}")
    children = TreeNode.objects.filter(parent=node)
    for child in children:
        get_tree(child, depth + 1)

root_node = TreeNode.objects.get(name='Root')
get_tree(root_node)
