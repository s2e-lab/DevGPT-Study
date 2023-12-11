def DFS_helper_scc(v, SCC):
    v.status = 'inprogress'
    SCC.append(v)  # Add the node to the SCC
    for w in v.getOutNeighbors():
        if w.status == 'unvisited':
            DFS_helper_scc(w, SCC)

# ...

for v in stack:
    if v.status == 'unvisited':
        SCC = []
        DFS_helper_scc(v, SCC)  # Call the modified DFS helper
        SCCs.append(SCC)
