import cv2
import numpy as np
from PIL import ImageGrab
from pathlib import Path
import time

class InventoryAnalyzer:
    def __init__(self, item_sprites_folder):
        self.item_sprites = {}
        self.load_sprites(item_sprites_folder)

    def load_sprites(self, sprites_folder):
        """
        Loads all item sprites from the dataset folder
        """
        sprites_path = Path(sprites_folder)
        print("Loading item templates.")
        loaded_count = 0

        for category in sprites_path.iterdir():
            if category.is_dir():
                for image in category.glob("*.png"):
                    sprite = cv2.imread(str(image))
                    if sprite is not None:
                        # Use filename without extension as item name
                        item_name = image.stem
                        self.item_sprites[item_name] = sprite
                        loaded_count += 1
        
        print(f"Loaded {loaded_count} item sprites.")

    def capture_inv_section(self):
        """
        Captures just the inventory section of the screen (top left by default)
        """
        inv_box = (15, 15, 360, 190) # Values could be wrong, check later
        screenshot = ImageGrab.grab(bbox=inv_box)
        return np.array(screenshot)
    
    def detect_items(self, inv_img):
        detected_items = []
        for item_name, sprite in self.item_sprites.items():
            result = cv2.matchTemplate(inv_img, sprite, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.85)

            for pt in zip(*locations[::-1]):
                detected_items.append({
                    'item' : item_name,
                    'position' : pt,
                    # Maybe add a slot member variable here?
                })
        return detected_items
    
    def get_inv_state(self):
        inv_img = self.capture_inv_section()
        items = self.detect_items(inv_img)
        return items

# Simple test without the full script
analyzer = InventoryAnalyzer("C:/Users/Sam/Documents/Comp Sci/Terraria Bot/Terraria-Bot/dataset")

items = analyzer.get_inv_state()
print(f"Found {len(items)} items:")
for item in items:
    print(f"  - {item['item']} at {item['position']}")