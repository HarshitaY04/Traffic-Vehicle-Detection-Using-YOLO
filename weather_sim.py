import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load your trained model
model = YOLO(r'runs\detect\train2\weights\best.pt')

def add_heavy_rain(image):
    """Simulates heavy Pune monsoon rain"""
    rain_img = image.copy()
    # Generate random rain drops
    drops = np.random.randint(0, 500, (800, 2)) # 800 raindrops
    for (x, y) in drops:
        # Draw slanted lines for falling rain
        cv2.line(rain_img, (x, y), (x + 3, y + 15), (200, 200, 200), 1)
    
    # Add a slight blur to simulate lower visibility
    rain_img = cv2.GaussianBlur(rain_img, (3, 3), 0)
    return rain_img

# 2. Point this to your BEST test image (like the close-up rickshaw)
image_path = r'Pune_Traffic_Fix-2\test\images\WhatsApp-Image-2026-03-15-at-9-15-52-PM-1-_jpeg.rf.9d6dd67bcc9f1d48dd010278d52292e1.jpg'
original_img = cv2.imread(image_path)

# 3. Apply the rain
rainy_img = add_heavy_rain(original_img)

# 4. Run YOLO on the rainy image
print("Detecting vehicles in the rain...")
results = model(rainy_img, conf=0.80)

# 5. Display the result
results[0].show()

# Optional: Save the rainy image for your PPT
cv2.imwrite('rainy_detection_result.jpg', results[0].plot())