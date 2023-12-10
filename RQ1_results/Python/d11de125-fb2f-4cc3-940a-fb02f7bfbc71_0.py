from PIL import Image

def enlarge_image(image_path, new_width, new_height):
    # Open the image
    image = Image.open(image_path)

    # Calculate the width and height difference
    width, height = image.size
    width_diff = new_width - width
    height_diff = new_height - height

    # Create a new blank image with the desired size
    new_image = Image.new("RGB", (new_width, new_height), "white")

    # Calculate the left padding
    left_padding = width_diff

    # Paste the original image onto the new image with padding
    new_image.paste(image, (left_padding, 0))

    # Save the enlarged image
    new_image.save("enlarged_icon.png")

    print("Image enlarged and saved as 'enlarged_icon.png'.")

# Specify the path to the original image and the desired new dimensions
image_path = "icon.png"
new_width = 225
new_height = 225

# Call the function to enlarge the image
enlarge_image(image_path, new_width, new_height)
