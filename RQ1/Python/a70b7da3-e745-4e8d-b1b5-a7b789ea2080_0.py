import cv2

# Read an image in BGR format (default color space for OpenCV)
image_bgr = cv2.imread('input_image.jpg')

# Convert BGR to RGB (OpenCV reads images in BGR format by default)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert BGR to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Convert BGR to LAB
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

# Convert BGR to YUV
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)

# Convert RGB back to BGR
image_bgr_converted = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# Display the original and converted images
cv2.imshow('BGR Image', image_bgr)
cv2.imshow('RGB Image', image_rgb)
cv2.imshow('HSV Image', image_hsv)
cv2.imshow('LAB Image', image_lab)
cv2.imshow('YUV Image', image_yuv)
cv2.imshow('Converted BGR Image', image_bgr_converted)

cv2.waitKey(0)
cv2.destroyAllWindows()
