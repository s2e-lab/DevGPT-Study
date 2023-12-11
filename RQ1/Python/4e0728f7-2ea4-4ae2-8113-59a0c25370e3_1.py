# Move to the top-left corner before drawing: 
turtle.penup() 
turtle.backward(size // 2)  # Move to the left
turtle.left(90) 
turtle.forward(size // 2)  # Move upwards
turtle.right(90)  # Now facing the right direction to start drawing
turtle.pendown()
