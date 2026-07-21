from ultralytics import YOLO
import cv2
import os

# 1. Load YOUR custom-trained model
model = YOLO('runs/detect/train/weights/best.pt') 

input_image = 'dataset_images/YOUR_IMAGE_NAME.jpg' # Pick one of your original images
output_folder = 'custom_test_results'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Run detection (Notice we don't need class filters anymore!)
results = model.predict(source=input_image, save=False, conf=0.25)

# 3. Save the result
annotated_frame = results[0].plot()
image_name = os.path.basename(input_image)
cv2.imwrite(os.path.join(output_folder, f"custom_{image_name}"), annotated_frame)

print(f"Check the '{output_folder}' folder to see if it identified the Auto-rickshaws!")