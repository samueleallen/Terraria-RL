import cv2 as cv
import numpy as np
import pytesseract

# def measure_health(healthbar_img):
#     """
#     Takes in a cropped image below an enemy detected and uses contour detection
#     to find the healthbar. If a healthbar is found, we then measure the ratio
#     of how filled the healthbar is and then determine health remaining.
#     """
#     # Convert img to grayscale format
#     img_gray = cv.cvtColor(healthbar_img, cv.COLOR_BGR2GRAY)
#     # Apply binary thresholding
#     ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)

#     # Detect contours on binary image
#     contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#     if contours:
#         # Get bounding box of largest contour detected
#         c = max(contours, key=cv.contourArea)
#         x, y, w, h = cv.boundingRect(c)  
#         cv.rectangle(healthbar_img, (x, y), (x+w, y+h), (255, 0, 0), 4) # draws blue rectangle for contour detection debugging
#         ratio = w / 45 # Healthbar is 45 pixels wide when on full screen 2560x1440 Resolution. For accurate health measurements change this line if resolution is different
#         return min(ratio, 1.0)
#     else:
#         return 0

def measure_enemy_health(healthbar_img, max_healthbar_width=45, debug=True):
    """
    Takes in a cropped image below an enemy detected and uses contour detection
    to find the healthbar. If a healthbar is found, we then measure the ratio
    of how filled the healthbar is and then determine health remaining.

    Uses the maximum value across all color channels as grayscale
    to ensure that orange/red health bars stay bright enough for contour detection.
    """
    # Extract channels
    b, g, r = cv.split(healthbar_img)
    
    # Take maximum value across channels for each pixel
    max_channel = np.maximum(np.maximum(r, g), b)
    
    # Apply binary thresholding and then detect contours on binary image
    ret, thresh = cv.threshold(max_channel, 150, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        
        if debug:
            # If debugging is on, draw rectangle for visualization of healthbar detection
            cv.rectangle(healthbar_img, (x, y), (x+w, y+h), (0, 0, 255), 4)
        
        ratio = w / max_healthbar_width # Healthbar is 45 pixels wide when on full screen 2560x1440 Resolution. For accurate health measurements change this line if resolution is different
        return min(ratio, 1.0)
    else:
        return 0
    
def measure_player_health(player_health_img):
    """
    Receives a screenshot of player's life and uses image processing to read the text value. Returns
    the ratio of health left as a percentage.
    
    Note: This setup requires you to have your health and mana style set to 'Fancy 2' in the settings. Additionally,
    it expects for a 2560x1440 resolution. Implementation for other resolutions has not yet been built.
    """
    player_health = 0

    return player_health
