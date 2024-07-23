import os
import cv2
import time
import torch
import numpy as np
from PIL import Image

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
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "Loadframe"
    CATEGORY = "VideoPlayer"

    def Loadframe(self, frame, frameRate, path):
        try:
            with open((path + str(frame) + ".txt"), 'r', encoding='utf-8') as file:
                content = file.read()
                if frameRate != 0:
                    time.sleep(1/int(frameRate))
            return (content,)
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
   
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_jpg_frame"
    CATEGORY = "VideoPlayer"

    def load_jpg_frame(self, frame, frameRate, path):
        try:
            image_path = os.path.join(path, f"{frame:05d}.jpg")
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                image = np.array(img).astype(np.float32) / 255.0
                #Convert to PyTorch tensor and add batch dimension
                image = torch.from_numpy(image)[None,]
            if frameRate != 0:
                time.sleep(1/int(frameRate))
            return (image,)
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
   
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "LoadVideoFrame"
    CATEGORY = "VideoPlayer"

    def LoadVideoFrame(self, video_path, frame, frameRate):
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)  # Subtract 1 because frame count starts at 0
            ret, img = cap.read()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            image = np.array(pil_image).astype(np.float32) / 255.0
            #Convert to PyTorch tensor and add batch dimension
            image_tensor = torch.from_numpy(image)[None,]
            if frameRate != 0:
                time.sleep(1/int(frameRate))
            cap.release()
            return (image_tensor,)
        except Exception as e:
            raise RuntimeError(f"Error reading video frame: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "LoadFrame": LoadFrame,
    "LoadJPGFrame": LoadJPGFrame,
    "LoadVideoFrame": LoadVideoFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFrame": "LoadFrame",
    "LoadJPGFrame": "LoadJPGFrame",
    "LoadVideoFrame": "Load Video Frame",
}