import time
class LoadFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}, {"forceInput": True}),
                "frameRate": ("INT", {"default": 0, "min": 0, "max": 144, "step": 1}),
                "path": ("STRING", {"forceInput": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "Loadframe"
    CATEGORY = "BadApple"

    def Loadframe(self, frame, frameRate, path):
        try:
            with open((path + str(frame) + ".txt"), 'r', encoding='utf-8') as file:
                content = file.read()
                if frameRate != 0:
                    time.sleep(1/int(frameRate))
            return (content,)
        except Exception as e:
            raise RuntimeError(f"Error reading file: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "LoadFrame": LoadFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFrame": "LoadFrame",
}