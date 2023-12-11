prediction_gif_path = model_ai.predict(image_path)[0]  # Obțineți calea către fișierul GIF returnat
prediction_image = imageio.imread(prediction_gif_path)
