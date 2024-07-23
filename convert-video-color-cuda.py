import cv2
import numpy as np
import os
import cupy as cp
from numba import cuda

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

COLOR_ARRAY = np.array(list(COLORS.values()), dtype=np.uint8)
EMOJI_LIST = list(COLORS.keys())

@cuda.jit
def get_closest_emoji_kernel(rgb_image, color_array, result):
    x, y = cuda.grid(2)
    if x < rgb_image.shape[0] and y < rgb_image.shape[1]:
        min_distance = 2147483647  # Max value for int32
        min_index = 0
        for i in range(color_array.shape[0]):
            distance = 0
            for c in range(3):
                diff = int(rgb_image[x, y, c]) - int(color_array[i, c])
                distance += diff * diff
            if distance < min_distance:
                min_distance = distance
                min_index = i
        result[x, y] = min_index

def image_to_emoji(image, width):
    height = int(image.shape[0] * width / image.shape[1])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
   
    d_image = cuda.to_device(resized)
    d_color_array = cuda.to_device(COLOR_ARRAY)
    d_result = cuda.device_array((height, width), dtype=np.int32)
   
    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(height / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(width / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
   
    get_closest_emoji_kernel[blocks_per_grid, threads_per_block](d_image, d_color_array, d_result)
   
    result = d_result.copy_to_host()
    emoji_array = np.array(EMOJI_LIST)[result]
   
    return '\n'.join(''.join(row) for row in emoji_array)

def video_to_emoji(video_path, output_folder, width):
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)
   
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        frame_count += 1
       
        emoji_frame = image_to_emoji(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width)
       
        with open(os.path.join(output_folder, f"{frame_count}.txt"), "w", encoding="utf-8") as f:
            f.write(emoji_frame)
       
        print(f"Processed frame {frame_count}")
   
    video.release()
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