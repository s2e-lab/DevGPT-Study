import time
import cv2
import numpy as np
import pyautogui

# Load the image template
template = cv2.imread('template.png', 0)
w, h = template.shape[::-1]

while True:
    # Capture screenshot
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Apply template Matching with a method based on the cross-correlation, normalized
    res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum. Else take maximum
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Define the threshold for detection. If max_val is greater than the threshold, then we have a match
    threshold = 0.9
    if max_val > threshold:
        center_loc = (top_left[0] + int(w / 2), top_left[1] + int(h / 2))  # Calculate center of the found image
        pyautogui.click(center_loc)  # Perform click action
        time.sleep(10)  # Wait for 10 seconds
