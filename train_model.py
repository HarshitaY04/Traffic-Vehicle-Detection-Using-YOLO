from ultralytics import YOLO

# 1. Load the "last" saved state instead of the base model
# Check your sidebar: the folder might be 'train', 'train2', etc. 
# Make sure the path below matches your actual folder name in 'runs/detect/'
model = YOLO('runs/detect/train/weights/last.pt') 

# 2. Resume the training
# Setting resume=True tells YOLO to pick up exactly where it crashed
model.train(resume=True)