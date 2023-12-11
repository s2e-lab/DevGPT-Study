import numpy as np

img = np.fromfile(dph_files[0], dtype=np.uint16)
img_size = img.size

target_sizes = {
    64000: (320, 200),
    256000: (640, 400)
}

if img_size in target_sizes:
    width, height = target_sizes[img_size]
else:
    # Calculate width and height based on the image size
    # You can adjust this calculation according to your requirements
    ratio = img_size / 64000  # Adjust the divisor as needed
    width = int(320 * ratio)   # Adjust the base width as needed
    height = int(200 * ratio)  # Adjust the base height as needed

print(f"Width: {width}, Height: {height}")
