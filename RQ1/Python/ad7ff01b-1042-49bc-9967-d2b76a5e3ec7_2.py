class Brick:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def draw(self, window):
        pygame.draw.rect(window, (255, 255, 255), pygame.Rect(self.x, self.y, self.w, self.h))
