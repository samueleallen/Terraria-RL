import cv2
import numpy as np
import pyautogui
import time

# Takes screenshot
screenshot = pyautogui.screenshot()

# Convert screesnhot to form cv2 can use
cv2_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Show screenshot
cv2.imshow('Terraria', cv2_img)

# cv2.waitKey(0)
cv2.destroyAllWindows