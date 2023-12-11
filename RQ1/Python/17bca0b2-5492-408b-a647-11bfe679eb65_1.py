class Node:
    def __init__(self, v):
        self.value = v
        self.inNeighbors = []
        self.outNeighbors = []
        self.status = "unvisited"
        self.estD = np.inf
        # Remove or comment out the following line:
        # self.index = None
