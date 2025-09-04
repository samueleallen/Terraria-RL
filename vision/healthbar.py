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
        self.fc1_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 11)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
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

    Filters out any detections that don't match possible healthbar shapes.

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
    
    valid_contours = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)

        # Apply filtering
        if 1 <= w <= 40 and 10<= h <= 12: # need to change these values based on resolution
            valid_contours.append((contour, x, y, w, h))
        
    if valid_contours:
        # Pick largest by area among valid contours
        c, x, y, w, h = max(valid_contours, key=lambda item:cv.contourArea(item[0]))

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
    ret, thresh = cv.threshold(gray_img, 185, 255, cv.THRESH_BINARY)

    # Detect any contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    char_images = []
    # Sort each contour from left to right
    contours = sorted(contours, key=lambda c: cv.boundingRect(c)[0])

    for c in contours:
        x, y, w, h = cv.boundingRect(c)

        char_img = life_nums[y:y+h, x:x+w]
        char_images.append(char_img)
        if debug:
            cv.rectangle(life_nums, (x,y), (x+w, y+h), (0, 255, 0), 1)

    return char_images

def filter_char_imgs(char_images):
    """
    Inputs: A list containing preprocessed screenshots ready for CNN of predicted locations of the chararacters.
    
    Case 1: 5 contours detected, 5 chars predicted. 
        A: This means we need to split our 2nd char img roughy in half and predict each side as different chars. The right should be a slash, and the left should be a number
        
    Case 2: 4 Contours detected, 4 chars predicted.
        A: This means we need to split our first char img in half and predict each side as different chars, should be same output as case 1. There has to be a minimum of 5 chars
        so this case is guranteed to have an error.
    """
    initial_images = char_images # Backup images for any changes we make
    length = len(char_images)
    # print("Length of Char Images List: ", length)
    split_imgs = []

    # print([img.shape for img in char_images])

    if length == 4:
        # Get height and width of image
        height, width = char_images[0].shape[:2] # It has channel/batch dimension first so we need to extract 2nd and 3rd dimension of shape. (1, 28 28)

        # Split image in half (may need to adjust)
        split_imgs = [
                    char_images[0][:, :width//2], # left half
                    char_images[0][:, width//2:] # right half
                    ]

        # Pop out first element/image from char images list and add the rest to fixed_images
        char_images.pop(0)

        char_images[0:0] = split_imgs

    if length == 5:
        # First, have to check if a slash was detected, if so, we are fine. (2nd image should be a slash in this case)
        # Else, we should split second image.
        # Predict each character
        class_names = [str(i) for i in range(10)] + ["/"]

        pred_char = ""
        char_tensor = torch.tensor(preprocess_img(char_images[1]), dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(char_tensor)
        preds = torch.softmax(outputs, dim=1)
        class_index = torch.argmax(preds, dim=1).item()
        label = class_names[class_index]
        pred_char += label

        if pred_char != "/":

            height, width = char_images[0].shape[:2]

            print(height, width)

            # Split image in half (may need to adjust)
            split_imgs = [
                        char_images[0][:, :width//2], # left half
                        char_images[0][:, width//2:] # right half
                        ]

            # Pop out second element/image from char images list and add the rest to fixed_images
            char_images.pop(0)

            char_images[0:0] = split_imgs

                # From here, we must check if either split image is a slash, if not, we messed up and need to use backup list of images and split 1st image instead of 2nd.
                # for i in range(len(char_images)):
                #     pred_string = ""
                #     char_tensor = torch.tensor(preprocess_img(char_images[1]), dtype=torch.float32).unsqueeze(0).to(device)
                #     outputs = model(char_tensor)
                #     preds = torch.softmax(outputs, dim=1)
                #     class_index = torch.argmax(preds, dim=1).item()
                #     label = class_names[class_index]
                #     pred_char += label

    for i in range(len(char_images)):
        char_images[i] = preprocess_img(char_images[i])

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

health_history = []
N = 3

def filter_health_preds(new_health):

    new_health = min(new_health, 100)

    # Compare to last N health values
    if len(health_history) < N:
        health_history.append(new_health)
        print(health_history)
        return new_health

    # If the new value is inconsistent with previous ones, we ignore it
    avg_recent = sum(health_history[-N:]) / N
    if abs(new_health - avg_recent) > 20:
        print("abs value is true, return last val")
        # Reject the prediction and return last known value
        return health_history[-1]

    health_history.append(new_health)
    print("returning passed health")
    return new_health
        

def measure_player_health(player_health_img):
    """
    Receives a screenshot of the game and uses a custom-built CNN to classify digits representing player health.
    
    Note: This setup requires you to have your health and mana style set to 'Fancy 2' in the settings. Additionally,
    it expects for a 2560x1440 resolution. Implementation for other resolutions has not yet been built.
    """

    class_names = [str(i) for i in range(10)] + ["/"]
    char_imgs = find_life_nums(player_health_img)
    # print("Before save function call, char img length: ", len(char_imgs))
    char_imgs = filter_char_imgs(char_imgs)

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
            curr = filter_health_preds(curr)
            health_percent = curr / max

    return health_percent