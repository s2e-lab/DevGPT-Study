from screeninfo import get_monitors

def get_screen_center():
    monitor = get_monitors()[0]  # Assuming the first monitor is the primary
    center_x = monitor.width // 2
    center_y = monitor.height // 2
    return center_x, center_y

if __name__ == "__main__":
    center_x, center_y = get_screen_center()
    print(f"Screen Center: ({center_x}, {center_y})")
