import numpy as np
import cv2 as cv
from PIL import ImageGrab
import pygetwindow as gw
from time import sleep
import ctypes
import os

"""
Useful script for data collection. Loops over game, taking screenshots roughly every second
and saves screenshots to a directory. 

Note: It will overwrite the last screenshots every time you reuse the program
"""

ctypes.windll.user32.SetProcessDPIAware()

# Edit save directory as needed
save_directory = "C:/Users/Sam/Documents/Comp Sci/Terraria Bot/Terraria-Bot/dataset/game_screenshots/night_time"
counter = 0

paused = False
while(True):
    try:

        counter += 1

        # Create directory for screenshots to go to if it doesn't already exist
        os.makedirs(save_directory, exist_ok=True)

        filename = f"screenshot_{counter}.png"

        full_path = os.path.join(save_directory, filename)

        key = cv.waitKey(1)
        if key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        if key == ord('q'):
            cv.destroyAllWindows()
            break
        if paused:
            cv.waitKey(1)
            continue
        # Get Terraria window
        window = gw.getWindowsWithTitle('Terraria:')[0]

        # Get coords of window so we can grab image
        left, top, right, bottom = window.left, window.top, window.right, window.bottom
        
        # Capture specific region defined by window's coords
        screenshot = ImageGrab.grab(bbox=(left,top,right,bottom))

        # Save screenshot to specified path        
        screenshot.save(full_path)

        # Format screenshot for cv image show
        screenshot = np.array(screenshot)

        cv.imshow('Terraria Object Detection', screenshot)

        sleep(1)

        # press 'q' with the output window focused to exit program
        # waits 1ms every loop to process key inputs
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break
    except IndexError: 
        print("Error: Couldn't find Terraria window")
        break
    except Exception as e:
        print(f"Error: {e}")
    
print("Done.")