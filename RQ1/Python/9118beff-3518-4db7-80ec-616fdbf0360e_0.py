math = SCCNode("Mathematics")
physics = SCCNode("Physics")
history = SCCNode("History")
art = SCCNode("Art")
biology = SCCNode("Biology")
chemistry = SCCNode("Chemistry")

G = SCCGraph()
V = [math, physics, history, art, biology, chemistry]
for v in V:
    G.addVertex(v)
    
# Edges forming three SCCs
# SCC 1: Math and Physics
# SCC 2: History and Art
# SCC 3: Biology and Chemistry
E = [ (math, physics), (physics, math), 
      (history, art), (art, history), 
      (biology, chemistry), (chemistry, biology) ]
for x, y in E:
    G.addDiEdge(x, y)

print(G)
