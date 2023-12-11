class Player:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def make_move(self, board):
        return self.algorithm.best_move(board, self)
