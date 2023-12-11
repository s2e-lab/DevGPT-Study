function UCB1(node):
    if node.visit_count == 0:
        return infinity  # Ensure unvisited nodes are selected first
    
    exploitation_term = node.total_score / node.visit_count
    exploration_term = sqrt(2 * log(node.parent.visit_count) / node.visit_count)
    return exploitation_term + exploration_term
