from ultralytics import YOLO
import os
import shutil

# 1. Load the YOLO-World Zero-Shot Model
# This will automatically download the YOLO-World AI the first time you run it
model = YOLO('yolov8s-world.pt')

# 2. Define the exact Indian vehicles you want to find!
# The AI will read these words and look for them in the images.
custom_classes = ["auto rickshaw", "car", "bus", "truck", "motorcycle", "three wheeler passenger vehicle"]
model.set_classes(custom_classes)

input_folder = 'dataset_images'

# 3. Set up the exact folder structure needed for custom YOLO training later
output_images = 'custom_dataset/images/train'
output_labels = 'custom_dataset/labels/train'

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

print(f"Starting Auto-Labeling for: {custom_classes}")

# 4. Loop through all your images and let the AI do the work
for image_name in os.listdir(input_folder):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, image_name)
        
        # Lower confidence to catch hidden vehicles in dense Pune traffic
        results = model.predict(image_path, conf=0.05) 
        
        # Copy the image into the new training folder
        shutil.copy(image_path, os.path.join(output_images, image_name))
        
        # Save the YOLO bounding box coordinates as a text file!
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        txt_path = os.path.join(output_labels, txt_name)
        
        # This automatically writes the YOLO format label file for you
        results[0].save_txt(txt_path)

print("\nSuccess! Auto-labeling complete. Your dataset is prepared.")