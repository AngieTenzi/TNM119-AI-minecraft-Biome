import os
from PIL import Image

# Input and output folders
input_folder = "pics"
output_folder = "resized_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Target size
target_size = (320, 180)

# Supported image formats
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(input_path) as img:
                # Resize image
                resized_img = img.resize(target_size)

                # Save to output folder
                resized_img.save(output_path)

                print(f"Resized: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")