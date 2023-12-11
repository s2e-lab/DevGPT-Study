import random

BOARD_SIZE = 15
EMPTY = '.'
PLAYER = 'X'
AI = 'O'

def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

def is_win(board, x, y, player):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 0
        for i in range(-4, 5):
            nx, ny = x + i * dx, y + i * dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == player:
                count += 1
                if count == 5:
                    return True
            else:
                count = 0
    return False

def ai_move(board, difficulty):
    if difficulty == 'easy':
        while True:
            x, y = random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1)
            if board[x][y] == EMPTY:
                return x, y
    elif difficulty == 'medium':
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x][y] == EMPTY:
                    board[x][y] = AI
                    if is_win(board, x, y, AI):
                        return x, y
                    board[x][y] = PLAYER
                    if is_win(board, x, y, PLAYER):
                        return x, y
                    board[x][y] = EMPTY
        return ai_move(board, 'easy')

    # For a 'hard' AI, you might want to use Minimax or other algorithms. For simplicity, it's omitted here.

def play_game(difficulty='medium'):
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    print_board(board)

    while True:
        x, y = map(int, input("Enter your move (row column): ").split())
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == EMPTY:
            board[x][y] = PLAYER
            print_board(board)
            if is_win(board, x, y, PLAYER):
                print("You win!")
                return

            print("AI is thinking...")
            x, y = ai_move(board, difficulty)
            board[x][y] = AI
            print_board(board)
            if is_win(board, x, y, AI):
                print("AI wins!")
                return

if __name__ == "__main__":
    difficulty = input("Choose difficulty (easy/medium): ").lower()
    play_game(difficulty)
