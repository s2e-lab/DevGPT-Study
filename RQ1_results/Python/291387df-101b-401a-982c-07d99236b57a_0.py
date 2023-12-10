if solved:
    displayBoard(board)
    for move in solutionMoves:
        print('Move', move)
        makeMove(board, move)
        print() # Print a newline.
        displayBoard(board)
