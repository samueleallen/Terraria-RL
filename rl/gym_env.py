import sys
import os

# Get the absolute path of the parent directory of the current script's directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add this parent directory to the system path.
sys.path.append(parent_dir)

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import numpy as np
import pydirectinput
import numpy as np
import cv2 as cv
from PIL import ImageGrab
import pygetwindow as gw
import time
import ctypes
from ultralytics import YOLO
from vision import healthbar
from vision.healthbar import ProcessingError
import math

ctypes.windll.user32.SetProcessDPIAware() # Set program to be dpi aware, it can mess with screenshots if not
model = YOLO('runs/detect/train5/weights/best.pt') # Adjust path as you update object detection model

class TerrariaEnv(Env):
    def __init__(self):
        self.observation_space = Box(low=0, high=1, shape=(4*4+1,)) # This includes the 4 nearest enemy positions (x and y), enemy health, enemy distance, and player health
        self.actions = ["move_left", "move_right", "jump", "attack", "do_nothing"] # 5 actions: left, right, jump, attack, stay still
        self.action_space = Discrete(len(self.actions)) 
        self.previous_enemy_healths = [0, 0 , 0, 0] # List to store previous healths of nearest 4 enemies
        self.previous_player_health = 0
        self.consecutive_zero_health = 0

    def get_observation(self):
        # Get Terraria window
        try:
            window = gw.getWindowsWithTitle('Terraria:')[0]
            # Get coords of window so we can grab image
            left, top, right, bottom = window.left, window.top, window.right, window.bottom
            
            # Capture specific region defined by window's coords
            screenshot = ImageGrab.grab(bbox=(left,top,right,bottom))
            
            # Convert screenshot to acceptable format
            screenshot = np.array(screenshot)
            screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR) # converts from RGB to BGR
        except IndexError: 
            print("Error: Couldn't find Terraria window")
            return None, None
        
        enemy_data = []

        # Run YOLO object detection
        results = model.predict(screenshot, conf=0.65, verbose=False)

        # Draw any detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                # Optionally draw boxes around detected enemies
                # TODO: Fix this
                cv.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Calculate enemy center
                enemy_center_x = (x1 + x2) / 2
                enemy_center_y = (y1 + y2) / 2

                # Calculate enemy distance to player
                distance = math.sqrt((enemy_center_x-1280)**2 + (enemy_center_y-720)**2)

                # Measure health on any enemies detected
                healthbar_crop = screenshot[(y2+5):(y2+25), (max(0, x1-5)):(min(screenshot.shape[1], x2+5))]

                # Optional line for debugging healthbar region. May need to check it out if resolution is not 2560x1440
                # cv.rectangle(screenshot, (max(0, x1-5), (y2+5)), ((min(screenshot.shape[1], x2+5)), (y2+25)), (0, 255, 0), 1)
                health = healthbar.measure_enemy_health(healthbar_crop)

                # Store enemy data
                enemy_data.append({
                    "box": box,
                    "distance": distance,
                    "health": health
                })

        # Find the nearest 4 enemies
        if enemy_data:
            sorted_enemies = sorted(enemy_data, key=lambda x: x['distance'])
            nearest_four_enemies = sorted_enemies[:4]
        else: 
            nearest_four_enemies = [] # Return empty list if no enemies found

        # Check player health
        try:
            player_health = healthbar.measure_player_health(screenshot)
            cv.imshow("Terraria", screenshot)
            cv.waitKey(1)
            return nearest_four_enemies, player_health
        except ProcessingError:
            print("Error processing player health, returning previous value")
            return nearest_four_enemies, self.previous_player_health
        
        
    
    def step(self, action_index):
        action_name = self.actions[action_index]

        # Get current observation
        nearest_four_enemies, player_health = self.get_observation()

        if action_name == "move_left":
            pydirectinput.keyDown('a')
            time.sleep(0.15)
            pydirectinput.keyUp('a')
        elif action_name == "move_right":
            pydirectinput.keyDown('d')
            time.sleep(0.15)
            pydirectinput.keyUp('d')  
        elif action_name == "jump":
            pydirectinput.press('space')
        elif action_name == "attack":
            if nearest_four_enemies:
                # Get nearest enemy
                nearest_enemy = nearest_four_enemies[0]

                # Get coordinates of nearest enemy
                x1, y1, x2, y2 = nearest_enemy['box']
                enemy_center_x = (x1 + x2) / 2
                enemy_center_y = (y1 + y2) / 2

                # Move cursor to enemy position and click
                pydirectinput.moveTo(int(enemy_center_x), int(enemy_center_y))
                
                # Have to hold click cause game doesn't recognize single click
                pydirectinput.mouseDown()
                time.sleep(0.2)
                pydirectinput.mouseUp()
            
            else:
                # Click randomly if no enemy is detected
                pydirectinput.mouseDown()
                time.sleep(0.2)
                pydirectinput.mouseUp()

        reward = self.calculate_reward(nearest_four_enemies, player_health)

        # Add penalty for not moving
        if action_name == "do_nothing":
            reward -= 0.1
        else:
            reward += 0.05

        if player_health <= 0:
            self.consecutive_zero_health += 1
            print("Player health said to be 0: ", player_health)
        else:
            self.consecutive_zero_health = 0

        terminated = (self.consecutive_zero_health >= 5)
        if terminated:
            print("Terminating program, 5 0s detected.")
            

        truncated = False # no set time limit for now

        # Format observation for RL model
        formatted_obs = np.zeros(self.observation_space.shape)

        # Iterate through enemies and add data to observation array
        for i, enemy in enumerate(nearest_four_enemies):
            formatted_obs[i*4] = enemy['box'][0] # x-coord
            formatted_obs[i*4+1] = enemy['box'][1] # y-coord
            formatted_obs[i*4+2] = enemy['distance'] # distance to player
            formatted_obs[i*4+3] = enemy['health'] # enemy health

        formatted_obs[-1] = player_health

        return formatted_obs, reward, terminated, truncated, {}
    
    def calculate_reward(self, enemies_data, player_health):
        reward = 0

        current_enemy_healths = [enemy['health'] for enemy in enemies_data[:4]]

        # Check for damage dealt to nearest 4 enemies
        for i in range(min(len(current_enemy_healths), len(self.previous_enemy_healths))):
            previous_health = self.previous_enemy_healths[i]
            current_health = current_enemy_healths[i]

            damage_dealt = previous_health - current_health

            if damage_dealt > 0:
                # Calculate exponential reward (incentivizes damaging lower health enemies)
                multiplier = 1 / (current_health + 0.1)
                # Add reward for damage dealth to enemy
                reward += damage_dealt * multiplier
                print(f"Adding reward of {damage_dealt * multiplier} for damaging enemy.")
        self.previous_enemy_healths = current_enemy_healths
        
        # Add a negative reward if taking damage
        if player_health < self.previous_player_health:
            reward -= 5
            print(f"Minusing reward of 5 for taking damage.")
        self.previous_player_health = player_health

        # Add smaller reward for staying close to enemy
        if enemies_data and enemies_data[0]['distance'] < 100:
            reward += 3
            print("Adding reward of 3 for staying close to enemy")

        return reward
    
    def reset(self, seed=None, options=None):
        print("Player died, resetting episode.")

        time.sleep(11) # Need to change the amount of time we sleep for if fighting a boss

        # Reset state variables
        self.previous_enemy_healths = [0, 0, 0, 0]
        self.previous_player_health = 1.0

        # Get initial observation for new episode
        nearest_enemies, player_health = self.get_observation()

        # Format observation again
        formatted_obs = np.zeros(self.observation_space.shape)
        if nearest_enemies:
            for i, enemy in enumerate(nearest_enemies):
                formatted_obs[i*4] = enemy['box'][0] # x-coord
                formatted_obs[i*4+1] = enemy['box'][1] # y-coord
                formatted_obs[i*4+2] = enemy['distance'] # distance to player
                formatted_obs[i*4+3] = enemy['health'] # enemy health

        formatted_obs[-1] = player_health

        info = {}

        return formatted_obs, info
    
    
        
