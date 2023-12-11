class ImmutableBoard:
    def __init__(self, dimensions):
        self.board = [['' for _ in range(dimensions)] for _ in range(dimensions)]

    def place_marker(self, x, y, marker):
        if self.board[y][x] != '':
            raise ValueError("Cell already occupied")
        
        # Create a new board instead of modifying the existing one
        new_board = [row.copy() for row in self.board]
        new_board[y][x] = marker

        return ImmutableBoard.from_existing_board(new_board)

    @classmethod
    def from_existing_board(cls, board_data):
        new_instance = cls(len(board_data))
        new_instance.board = board_data
        return new_instance
