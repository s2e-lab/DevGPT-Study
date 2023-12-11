import os

def write_tree(node, path='node', depth=0):
    if hasattr(node, 'points'):
        # this is a leaf node, so write the points to a file
        df = pd.DataFrame(node.points, columns=['x', 'y', 'z'])
        df.to_csv(f'{path}.csv', index=False)
    else:
        # this is an internal node, so recurse on the children
        for i, child in enumerate(node.children):
            write_tree(child, path=f'{path}_{i}', depth=depth+1)

# write the octree to files
write_tree(OctreeNode(points))
