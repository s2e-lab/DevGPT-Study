# Move the turtle to the top of the equilateral triangle:
height = size * math.sqrt(3) / 2
turtle.penup()
turtle.left(90) # Turn to face upward.
turtle.forward(height * (2/3)) # Move to the top corner.
turtle.right(150) # Turn to face the bottom-left corner.
turtle.pendown()
