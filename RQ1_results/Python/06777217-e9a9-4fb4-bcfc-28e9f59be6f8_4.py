# Creating a simple tree structure

root = TreeNode.objects.create(name='Root', path='root')
child1 = TreeNode.objects.create(name='Child 1', path='root.child1', parent=root)
child2 = TreeNode.objects.create(name='Child 2', path='root.child2', parent=root)
subchild1 = TreeNode.objects.create(name='Subchild 1', path='root.child1.subchild1', parent=child1)
