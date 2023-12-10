class Graph:
    def __init__(self):
        self.vertices = []
        self.next_index = 0 # Add this line to initialize the next available index

    def addVertex(self, n):
        n.index = self.next_index # Assign the next available index to the vertex
        self.next_index += 1 # Increment the next available index
        self.vertices.append(n)
