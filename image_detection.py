from ultralytics import YOLO
import cv2
import os

# 1. Load the pre-trained YOLO model (matches your PyCharm screenshot)
model = YOLO('yolo11n.pt') 

input_folder = 'dataset_images'
output_folder = 'filtered_results'

# Create an output folder for the finished images
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Starting detection... Filtering out 'person' and other non-vehicle classes.")

# 2. Loop through all 30 of your images
for image_name in os.listdir(input_folder):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, image_name)
        
        # 3. Predict ONLY vehicle classes (2: car, 3: motorcycle, 5: bus, 7: truck)
        results = model.predict(source=image_path, classes=[2, 3, 5, 7], save=False)
        
        # 4. Draw the boxes and save the new image
        annotated_frame = results[0].plot()
        save_path = os.path.join(output_folder, image_name)
        cv2.imwrite(save_path, annotated_frame)

print(f"Success! All images processed. Check the '{output_folder}' folder to see the results.")