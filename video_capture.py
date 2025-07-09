import numpy as np
import cv2 as cv
from PIL import ImageGrab
import pygetwindow as gw
from time import time
import ctypes

"""
Note: There can be a funny little issue where ImageGrab and pygetwindow will disagree
on where things are located on the screen if your DPI is not set to 100% so we will set
the program to be "DPI Aware" (ignore window scaling basically)
"""

ctypes.windll.user32.SetProcessDPIAware()

loop_time = time()
while(True):
    try:
        # Get Terraria window
        window = gw.getWindowsWithTitle('Terraria:')[0]

        # Get coords of window so we can grab image
        left, top, right, bottom = window.left, window.top, window.right, window.bottom
        
        # Capture specific region defined by window's coords
        screenshot = ImageGrab.grab(bbox=(left,top,right,bottom))
        
        # Convert screenshot to acceptable format
        screenshot = np.array(screenshot)
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR) # converts from RGB to BGR

        cv.imshow('Computer Vision', screenshot)

        print("FPS {}".format(1 / (time() - loop_time)))
        loop_time = time()

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