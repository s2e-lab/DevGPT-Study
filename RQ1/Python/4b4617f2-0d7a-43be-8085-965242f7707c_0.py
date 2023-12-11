from graphviz import Digraph

# Creating Nodes (assuming Node class is defined elsewhere)
A = Node('A')
B = Node('B')
C = Node('C')
D = Node('D')

# Adding Neighbors
A.addOutNeighbor(B, 2)
A.addOutNeighbor(C, 3)
B.addOutNeighbor(C, 1)
B.addOutNeighbor(D, 3)
C.addOutNeighbor(D, 2)

# Creating Graphviz object
dot = Digraph()

# Adding Edges for Original Graph
dot.edge('A', 'B', label='2')
dot.edge('A', 'C', label='3')
dot.edge('B', 'C', label='1')
dot.edge('B', 'D', label='3')
dot.edge('C', 'D', label='2')

# Render the graph
dot.render('graph.gv', view=True)
