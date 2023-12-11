class Node:
    def run(self):
        pass

class Selector(Node):
    def __init__(self, children):
        self.children = children

    def run(self):
        for child in self.children:
            if child.run() == "Success":
                return "Success"
        return "Failure"

class Sequence(Node):
    def __init__(self, children):
        self.children = children

    def run(self):
        for child in self.children:
            if child.run() == "Failure":
                return "Failure"
        return "Success"

class IsObstacleNearby(Node):
    def run(self):
        return "Success" if is_obstacle_nearby() else "Failure"

class MoveForward(Node):
    def run(self):
        move_forward()
        return "Success"

class StopMoving(Node):
    def run(self):
        stop_moving()
        return "Success"

# Constructing Behavior Tree
root = Selector([
    Sequence([IsObstacleNearby(), StopMoving()]),
    Sequence([MoveForward()])
])

# Main loop for BT
while True:
    root.run()
