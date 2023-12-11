# Find a node by its path
target_path = 'root.child1.subchild1'  # Specify the path you want to search for
node = TreeNode.objects.filter(path=Lquery(target_path)).first()

if node:
    print(f"Node found: {node.name}")
else:
    print("Node not found")
