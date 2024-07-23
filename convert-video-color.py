import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# Pre-compute color distances
COLORS = {
    '🟥': (255, 0, 0),    # Red
    '🟧': (255, 165, 0),  # Orange
    '🟨': (255, 255, 0),  # Yellow
    '🟩': (0, 255, 0),    # Green
    '🟦': (0, 0, 255),    # Blue
    '🟪': (128, 0, 128),  # Purple
    '🟫': (165, 42, 42),  # Brown
    '⬛': (0, 0, 0),      # Black
    '⬜': (255, 255, 255) # White
}

COLOR_ARRAY = np.array(list(COLORS.values()))
EMOJI_LIST = list(COLORS.keys())

def get_closest_emoji(rgb):
    distances = np.sum((COLOR_ARRAY - rgb) ** 2, axis=1)
    return EMOJI_LIST[np.argmin(distances)]

def image_to_emoji(image, width):
    height = int(image.shape[0] * width / image.shape[1])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
   
    vectorized_get_closest = np.vectorize(get_closest_emoji, signature='(n)->()')
    emoji_array = vectorized_get_closest(resized.reshape(-1, 3)).reshape(height, width)
   
    return '\n'.join(''.join(row) for row in emoji_array)

def process_frame(args):
    frame_number, frame, width, output_folder = args
    emoji_frame = image_to_emoji(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width)
    with open(os.path.join(output_folder, f"{frame_number}.txt"), "w", encoding="utf-8") as f:
        f.write(emoji_frame)
    return frame_number

def video_to_emoji(video_path, output_folder, width):
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)
   
    frame_count = 0
    frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        frame_count += 1
        frames.append((frame_count, frame, width, output_folder))
   
    video.release()
   
    with ThreadPoolExecutor() as executor:
        for processed_frame in executor.map(process_frame, frames):
            print(f"Processed frame {processed_frame}")
   
    print("Video processing completed")

# Usage
video_path = input("Enter video path: ")
try:
    width = int(input("Enter desired width (100 is usually best): "))
except:
    print("Invalid width, defaulting to 100")
    width = 100
output_folder = "./frames"
video_to_emoji(video_path, output_folder, width)