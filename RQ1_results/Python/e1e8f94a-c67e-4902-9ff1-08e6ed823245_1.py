class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

def add_child(parent, child):
    parent.children.append(child)

# Create the tree from the visualization:
root = Node("A")

B = Node("B")
C = Node("C")
D = Node("D")

add_child(root, B)
add_child(root, C)
add_child(root, D)

E = Node("E")
F = Node("F")
add_child(B, E)
add_child(B, F)

G = Node("G")
add_child(C, G)

H = Node("H")
I = Node("I")
add_child(D, H)
add_child(D, I)

J = Node("J")
add_child(E, J)

K = Node("K")
add_child(G, K)
