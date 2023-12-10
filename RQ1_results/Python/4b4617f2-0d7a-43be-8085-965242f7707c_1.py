# Creating Nodes
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

# Creating Graph
G = [A, B, C, D]

# Running Prim's Algorithm
MST = prim(G, A)

# Expected Output: Edges (A, B), (B, C), and (C, D)
