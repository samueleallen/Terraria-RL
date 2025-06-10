import cv2
import numpy as np
import pyautogui
import time

def _calculate_reward(self):
    # Function to calculate rewards
    # +1 for staying alive
    # +10 for collecting an item (Have to detect inventory changes somehow)
    # -100 for dying (detect death screen)

    reward = 1 # basic survival reward

    # First, need to implement detections for item collection, death, and health changes. 
    # Then, maybe detections for basic daytime enemies like slimes.
