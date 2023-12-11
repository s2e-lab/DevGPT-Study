import cv2
import numpy as np

gif = cv2.VideoCapture(file.name)
frames = []
while True:
    ret, frame = gif.read()
    if not ret:
        break
    frames.append(frame)

# Procesați și modificați cadrele după necesitate
processed_frames = [...]
output_gif = cv2.VideoWriter('output.gif', cv2.VideoWriter_fourcc(*'GIF'), gif.get(cv2.CAP_PROP_FPS), (360, 360))
for frame in processed_frames:
    output_gif.write(frame)

output_gif.release()

with open('output.gif', 'rb') as f:
    gif_bytes = f.read()

return gif_bytes
