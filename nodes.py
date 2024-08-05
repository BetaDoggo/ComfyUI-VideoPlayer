import os
import cv2
import time
import torch
import shutil
import numpy as np
from PIL import Image
from comfy.utils import ProgressBar

class LoadFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "forceInput": True}),
                "frameRate": ("INT", {"default": 0, "min": 0, "max": 144, "step": 1}),
                "path": ("STRING", {"forceInput": False}),
            },
        }
   
    RETURN_TYPES = ("STRING", "FLOAT",)
    FUNCTION = "Loadframe"
    CATEGORY = "VideoPlayer"
    def Loadframe(self, frame, frameRate, path):
        try:
            path = path.replace('"', '') #make "copy as path" faster
            with open((path + str(frame) + ".txt"), 'r', encoding='utf-8') as file:
                content = file.read()
                timestamp = 0.00
                if frameRate != 0:
                    time.sleep(1/int(frameRate))
                    timestamp = frame/frameRate
            return (content, timestamp,)
        except Exception as e:
            raise RuntimeError(f"Error reading file: {str(e)}")

class LoadJPGFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "forceInput": True}),
                "frameRate": ("INT", {"default": 0, "min": 0, "max": 144, "step": 1}),
                "path": ("STRING", {"default": "", "forceInput": False}),
            },
        }
   
    RETURN_TYPES = ("IMAGE", "FLOAT",)
    FUNCTION = "load_jpg_frame"
    CATEGORY = "VideoPlayer"

    def load_jpg_frame(self, frame, frameRate, path):
        try:
            path = path.replace('"', '') #make "copy as path" faster
            image_path = os.path.join(path, f"{frame:05d}.jpg")
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                image = np.array(img).astype(np.float32) / 255.0
                #Convert to PyTorch tensor and add batch dimension
                image = torch.from_numpy(image)[None,]
            timestamp = 0.00
            if frameRate != 0:
                time.sleep(1/int(frameRate))
                timestamp = frame/frameRate
            return (image, timestamp,)
        except Exception as e:
            raise RuntimeError(f"Error reading image file: {str(e)}")

class LoadVideoFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "frame": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "forceInput": True}),
                "frameRate": ("INT", {"default": 0, "min": 0, "max": 144, "step": 1}),
            },
        }
   
    RETURN_TYPES = ("IMAGE", "FLOAT",)
    FUNCTION = "LoadVideoFrame"
    CATEGORY = "VideoPlayer"

    def LoadVideoFrame(self, video_path, frame, frameRate):
        try:
            video_path = video_path.replace('"', '') #make "copy as path" faster
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)  # Subtract 1 because frame count starts at 0
            ret, img = cap.read()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            image = np.array(pil_image).astype(np.float32) / 255.0
            #Convert to PyTorch tensor and add batch dimension
            image_tensor = torch.from_numpy(image)[None,]
            timestamp = 0.00
            if frameRate != 0:
                time.sleep(1/int(frameRate))
                timestamp = frame/frameRate
            cap.release()
            return (image_tensor, timestamp,)
        except Exception as e:
            raise RuntimeError(f"Error reading video frame: {str(e)}")

class ImageToEmoji:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 100, "min": 10, "max": 200, "step": 1}),
            },
        }
   
    RETURN_TYPES = ("STRING",)
    FUNCTION = "ImageToEmoji"
    CATEGORY = "VideoPlayer"

    def ImageToEmoji(self, image, width):
        #Convert from tensor to array
        image_np = 255. * image.cpu().numpy().squeeze()
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        img = Image.fromarray(image_np).convert('L')
        #resize
        aspect_ratio = img.height / img.width
        height = int(width * aspect_ratio)
        img = img.resize((width, height))
        pixels = np.array(img)
        #Set the black/white threshold to the mean value of the pixels
        threshold = np.mean(pixels)
        emoji_array = np.where(pixels > threshold, "⬜", "⬛")
        ascii_image = '\n'.join([''.join(row) for row in emoji_array])
        return(ascii_image,)
    
class AllInOnePlayer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "forceInput": True}),
                "video_path": ("STRING", {"forceInput": False}),
                "width": ("INT", {"default": 100, "min": 10, "max": 200, "step": 1}),
                "framerate": ("INT", {"default": 30, "min": 0, "max": 500, "step": 1}),
            },
        }
   
    RETURN_TYPES = ("STRING",)
    FUNCTION = "PlayVideo"
    CATEGORY = "VideoPlayer"

    def __init__(self):
        self.node_dir = os.path.dirname(os.path.abspath(__file__))

    def ImageToEmoji(self, image, width):
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        img = Image.fromarray(image)
        pixels = np.array(img)
        threshold = np.mean(pixels)
        emoji_array = np.where(pixels > threshold, "⬜", "⬛")
        ascii_image = '\n'.join([''.join(row) for row in emoji_array])
        return ascii_image

    def get_video_prefix(self, video_path):
        return os.path.splitext(os.path.basename(video_path))[0]

    def resize_frame(self, frame, target_width):
        height, width = frame.shape[:2]
        aspect_ratio = height / width
        new_height = int(target_width * aspect_ratio)
        return cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)

    def ExtractFrames(self, video_path, target_width):
        temp_frames_dir = os.path.join(self.node_dir, "temp_frames")
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        os.makedirs(temp_frames_dir)
        video_prefix = self.get_video_prefix(video_path)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #Initialize progress bar
        progress = ProgressBar(10)
        frame_count = 0
        next_progress_update = total_frames // 10  # Calculate frames per 10%
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = self.resize_frame(frame, target_width)
            emoji_frame = self.ImageToEmoji(resized_frame, target_width)
            
            frame_filename = os.path.join(temp_frames_dir, f"{video_prefix}_frame_{frame_count:04d}.txt")
            with open(frame_filename, 'w', encoding='utf-8') as f:
                f.write(emoji_frame)
           
            frame_count += 1
            #Update progress bar every 10% of frames processed
            if frame_count >= next_progress_update:
                progress.update(1)
                next_progress_update += total_frames // 10
        
        cap.release()

    def PlayVideo(self, frame, video_path, width, framerate):
        video_path = video_path.replace('"', '') #make "copy as path" faster
        progress = ProgressBar(10)
        progress.update
        temp_frames_dir = os.path.join(self.node_dir, "temp_frames")
        video_prefix = self.get_video_prefix(video_path)
       
        if not os.path.exists(temp_frames_dir) or not any(f.startswith(video_prefix) for f in os.listdir(temp_frames_dir)):
            self.ExtractFrames(video_path, width)

        frame_path = os.path.join(temp_frames_dir, f"{video_prefix}_frame_{frame:04d}.txt")
        if os.path.exists(frame_path):
            with open(frame_path, 'r', encoding='utf-8') as f:
                emoji_frame = f.read()
            
            if framerate != 0:
                time.sleep(1/framerate)
            return (emoji_frame,)
        else:
            return ("Failed to load frame. Either the video is over, the video path is wrong, or there's another error.",)

NODE_CLASS_MAPPINGS = {
    "LoadFrame": LoadFrame,
    "LoadJPGFrame": LoadJPGFrame,
    "LoadVideoFrame": LoadVideoFrame,
    "ImageToEmoji": ImageToEmoji,
    "AllInOnePlayer": AllInOnePlayer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFrame": "LoadFrame",
    "LoadJPGFrame": "LoadJPGFrame",
    "LoadVideoFrame": "Load Video Frame",
    "ImageToEmoji": "Image To Emoji",
    "AllInOnePlayer": "AllInOnePlayer",
}