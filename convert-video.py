import cv2
from PIL import Image
import numpy as np
import os

def image_to_emoji(image, width=100): #width can be increased to increase resolution but if it's too big it won't fit on screen
    img = Image.fromarray(image).convert('L')
    #resize based on width
    aspect_ratio = img.height / img.width
    height = int(width * aspect_ratio)
    img = img.resize((width, height))
    pixels = np.array(img)
    #set the black/white threshold to the mean value of the pixels
    threshold = np.mean(pixels)

    ascii_image = ""
    for row in pixels:
        for pixel in row:
            if pixel > threshold:
                ascii_image += "⬜"
            else:
                ascii_image += "⬛"
        ascii_image += "\n"
    
    return ascii_image

def video_to_ascii(video_path, output_folder, width=100):
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success: #if frame fails to load/doesn't exist
            break
        
        emoji_frame = image_to_emoji(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width)

        frame_count += 1
        with open(os.path.join(output_folder, f"{frame_count}.txt"), "w", encoding="utf-8") as f:
            f.write(emoji_frame)    
        print(frame_count)
    
    video.release()
    print("done")

# Usage
video_path = input("Enter video path: ")
output_folder = "./frames"
video_to_ascii(video_path, output_folder)