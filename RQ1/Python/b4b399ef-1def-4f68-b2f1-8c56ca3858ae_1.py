import pygame
import random

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Screen dimensions
WIDTH, HEIGHT = 640, 480

# Grid dimensions
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def move(self):
        head = self.body[0]
        new_head = ((head[0] + self.direction[0]) % GRID_WIDTH, 
                    (head[1] + self.direction[1]) % GRID_HEIGHT)
        self.body = [new_head] + self.body[:-1]

    def grow(self):
        head = self.body[0]
        new_head = ((head[0] + self.direction[0]) % GRID_WIDTH, 
                    (head[1] + self.direction[1]) % GRID_HEIGHT)
        self.body = [new_head] + self.body

    def collides_with_itself(self):
        return self.body[0] in self.body[1:]

    def draw(self, screen):
        for segment in self.body:
            pygame.draw.rect(screen, GREEN, 
                             (segment[0]*GRID_SIZE, segment[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

class Food:
    def __init__(self):
        self.position = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))

    def randomize_position(self):
        self.position = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))

    def draw(self, screen):
        pygame.draw.rect(screen, RED, 
                         (self.position[0]*GRID_SIZE, self.position[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake Game")
    clock = pygame.time.Clock()

    snake = Snake()
    food = Food()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and snake.direction != DOWN:
                    snake.direction = UP
                elif event.key == pygame.K_DOWN and snake.direction != UP:
                    snake.direction = DOWN
                elif event.key == pygame.K_LEFT and snake.direction != RIGHT:
                    snake.direction = LEFT
                elif event.key == pygame.K_RIGHT and snake.direction != LEFT:
                    snake.direction = RIGHT

        snake.move()

        if snake.body[0] == food.position:
            snake.grow()
            food.randomize_position()

        if snake.collides_with_itself():
            snake = Snake()
            food.randomize_position()

        screen.fill(BLACK)
        snake.draw(screen)
        food.draw(screen)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
