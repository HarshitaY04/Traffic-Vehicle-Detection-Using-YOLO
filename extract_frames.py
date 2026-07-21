import cv2
import os

def extract_frames(video_path, output_folder, extract_every_x_seconds=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:
        print("Error: Could not read the video. Please check if the file name is correct.")
        return
        
    frame_interval = fps * extract_every_x_seconds
    current_frame = 0
    saved_count = 0

    print(f"Opening video: {video_path}")
    print(f"Video FPS: {fps}. Extracting 1 frame every {extract_every_x_seconds} second(s)...")

    while True:
        success, frame = video.read()
        if not success:
            break
            
        if current_frame % frame_interval == 0:
            image_name = f"traffic_frame_{saved_count:04d}.jpg"
            save_path = os.path.join(output_folder, image_name)
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
        current_frame += 1

    video.release()
    print(f"Success! Extracted {saved_count} images to the '{output_folder}' folder.")

# --- SETTINGS ---
video_file = 'pune_traffic_video.mp4' 
output_directory = 'dataset_images'
extract_frames(video_file, output_directory, extract_every_x_seconds=1)