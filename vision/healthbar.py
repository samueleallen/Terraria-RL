import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*5*5, 64)
        self.fc2 = nn.Linear(64, 11)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device to gpu if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CNNModel()
model.load_state_dict(torch.load("text_classifier_weights.pth", map_location=device))
model.eval()
model.to(device)

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
    
def find_life_nums(img, debug=True):
    """
    Helper function that takes in the game screenshot and accurately finds
    where each number representing the player healthbar is located. This allows for
    future detection with our single-char cnn model.

    Returns a list of preprocessed images ready for cnn
    """
    life_nums = img[0:20, 2358:2440] # May need to adjust this rectangle

    # Convert img to grayscale
    gray_img = cv.cvtColor(life_nums, cv.COLOR_BGR2GRAY)
    # Apply binary thresholding
    ret, thresh = cv.threshold(gray_img, 125, 255, cv.THRESH_BINARY)

    # Detect any contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    char_images = []
    # Sort each contour from left to right
    contours = sorted(contours, key=lambda c: cv.boundingRect(c)[0])

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        char_img = life_nums[y:y+h, x:x+w]
        preprocessed = preprocess_img(char_img)
        char_images.append(preprocessed)

        if debug:
            cv.rectangle(life_nums, (x,y), (x+w, y+h), (0, 255, 0), 1)

    return char_images

    
def preprocess_img(img):
    """
    Helper function to preprocess an image to prepare for CNN model predictions
    """
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert img to grayscale

    resized_img = cv.resize(gray_img, (28, 28)) # Resize image to 28x28

    normalized_img = resized_img.astype("float32") / 255.0 # Normalize to [0, 1]

    normalized_img = np.expand_dims(normalized_img, axis=0) # Add batch dimension

    return normalized_img

def measure_player_health(player_health_img):
    """
    Receives a screenshot of the game and uses a custom-built CNN to classify digits representing player health.
    
    Note: This setup requires you to have your health and mana style set to 'Fancy 2' in the settings. Additionally,
    it expects for a 2560x1440 resolution. Implementation for other resolutions has not yet been built.
    """

    class_names = [str(i) for i in range(10)] + ["/"]
    char_imgs = find_life_nums(player_health_img)

    # Predict each character
    pred_str = ""
    for i, char_img in enumerate(char_imgs):
        char_tensor = torch.tensor(char_img, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(char_tensor)
        preds = torch.softmax(outputs, dim=1)
        class_index = torch.argmax(preds, dim=1).item()
        label = class_names[class_index]
        pred_str += label

    if pred_str:
        print("Prediction of player health: ", pred_str)

    health_percent = 0
    # Now parse through the health values to get health as a percentage
    if "/" in pred_str:
        # Split into current health vs max health
        curr, max = pred_str.split("/")
        if curr.isdigit() and max.isdigit():
            curr, max = int(curr), int(max)
            health_percent = curr / max
            
    return health_percent
