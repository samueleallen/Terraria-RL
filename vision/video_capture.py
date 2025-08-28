import numpy as np
import cv2 as cv
from PIL import ImageGrab
import pygetwindow as gw
from time import time
import ctypes
from ultralytics import YOLO
import healthbar

"""
Note: There can be a funny little issue where ImageGrab and pygetwindow will disagree
on where things are located on the screen if your DPI is not set to 100% so we will set
the program to be "DPI Aware" (ignore window scaling basically)
"""

ctypes.windll.user32.SetProcessDPIAware()

model = YOLO('runs/detect/train4/weights/best.pt') # Adjust path as you update object detection model

paused = False
loop_time = time()
while(True):
    try:
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
        
        # Convert screenshot to acceptable format
        screenshot = np.array(screenshot)
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR) # converts from RGB to BGR

        # Run YOLO object detection
        results = model.predict(screenshot, conf=0.65, verbose=False)

        # Draw any detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Measure health on any enemies detected
                healthbar_crop = screenshot[(y2+5):(y2+25), (max(0, x1-5)):(min(screenshot.shape[1], x2+5))]
                
                # Optional line for debugging healthbar region. May need to check it out if resolution is not 2560x1440
                # cv.rectangle(screenshot, (max(0, x1-5), (y2+5)), ((min(screenshot.shape[1], x2+5)), (y2+25)), (0, 255, 0), 1)
                health_ratio = healthbar.measure_enemy_health(healthbar_crop)

        # Check player health
        player_health = healthbar.measure_player_health(screenshot)

        cv.imshow('Terraria Object Detection', screenshot)

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


    