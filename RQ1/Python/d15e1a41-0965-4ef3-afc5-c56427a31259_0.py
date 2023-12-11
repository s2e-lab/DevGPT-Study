from PIL import Image
import os

def resize_images(directory):
    # Get all .jpg files in the directory
    image_files = [file for file in os.listdir(directory) if file.endswith('.jpg')]

    for file in image_files:
        # Open the image file
        image_path = os.path.join(directory, file)
        image = Image.open(image_path)

        # Resize the image
        resized_image = image.resize((640, 640))

        # Save the resized image, overwriting the original file
        resized_image.save(image_path)

        print(f"Resized {file}.")

# Specify the directory containing the images
directory_path = 'path/to/your/directory'

# Call the function to resize the images
resize_images(directory_path)
