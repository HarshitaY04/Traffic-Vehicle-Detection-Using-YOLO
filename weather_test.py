import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load your newly trained model
model = YOLO(r'C:\Users\Hii Harshita\OneDrive\Desktop\Pune_Traffic_Detection\runs\detect\train2\weights\best.pt')

def add_rain(image):
    # Simulates rain by adding thin white streaks
    rain_drops = np.random.randint(0, 500, (100, 2))
    for (x, y) in rain_drops:
        cv2.line(image, (x, y), (x + 2, y + 10), (200, 200, 200), 1)
    return image

# 2. Path to one of your test images
img_path = r'C:\Users\Hii Harshita\OneDrive\Desktop\Pune_Traffic_Detection\dataset_images\test_image.jpg' # Update this path!
img = cv2.imread(img_path)

# 3. Apply weather effect
rainy_img = add_rain(img.copy())

# 4. Run detection on rainy image
results = model(rainy_img)

# 5. Show results
results[0].show()