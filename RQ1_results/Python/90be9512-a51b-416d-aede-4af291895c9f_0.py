class RobotFSM:
    def __init__(self):
        self.state = "Idle"

    def transition(self):
        if self.state == "Idle":
            # Conditions for transition
            if is_obstacle_nearby():
                self.state = "Stop"
            else:
                self.state = "Move"
        
        elif self.state == "Move":
            # Move the robot
            move_forward()
            if is_obstacle_nearby():
                self.state = "Stop"
        
        elif self.state == "Stop":
            # Stop the robot
            stop_moving()
            if not is_obstacle_nearby():
                self.state = "Move"

# Main loop for FSM
robot = RobotFSM()
while True:
    robot.transition()
