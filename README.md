# Terraria Reinforcement Learning Agent
This project is a deep reinforcement learning agent built to autonomously play the game Terraria. The agent uses purely computer vision for data recognition and a custom-designed environment to learn how to perceive the game state, navigate, and engage in combat.

## Key Features
 * Real-time Perception: Utilizes a computer vision pipeline to process in-game screenshots and extract critical information.
 * Enemy Detection: Leverages a custom-trained Yolov8 model for real-time object detection of enemies.
 * Player Health Tracking: A custom Convolutional Neural Network (CNN) built with PyTorch is used to recognize digits on health bars, providing a precise measure of player health.
 * Enemy Health Tracking: Uses contour detection, measuring pixel intensity relative to bar width to estimate health in real-time.
 * Custom Environment: A gymnasium-compliant environment encapsulates the game logic, reward function, and observation space, making it compatible with standard DRL frameworks.
 * Autonomous Actions: The agent can move, jump, and attack based on its observations.

## Methodology
### Reinforcement Learning
The agent is trained using the **Proximal Policy Optimization** (PPO) algorithm, implemented via the `stable-baselines3` library. The model learns to optimize a custom reward function that incentivizes:
 * Dealing damage to enemies.
 * Surviving without taking damage.
 * Maintaining proximity to enemies for combat.

### Computer Vision
The vision pipeline is crucial for the agent's ability to see the game and includes:
 1. Screen Capture: Grabbing screenshots of the active Terraria window.
 2. Object Detection: Using a fine-tuned Yolov8 model to locate enemies with a custom dataset.
 3. Enemy Health Bar OCR: Cropping the health bar region below each detected enemy, measuring pixel intensity relative to the total width of the health bar.
 4. Player Health OCR: Cropping the player health region and using a PyTorch CNN to classify the health digits.
 5. Player Health: A dedicated function that reads and parses the player's health bar, filtering any misclassifications from the CNN.

## Technologies & Libraries
 * Python: Core programming language
 * PyTorch: For building and training the custom CNN for digit recognition.
 * Stable-Baselines3: Popular library for the PPO algorithm.
 * Gymnasium: For defining the environment's observation and action spaces.
 * OpenCV: For all image processing tasks.
 * YoloV8: For high-performance enemy detection.
 * Pydirectinput: For simulating keyboard and mouse inputs to control the game.
 * NumPy: For numerical operations and general data handling.

## Setup and Usage:
**Prerequisites**  
 * A python environment
 * The Terraria game running in a 2560x1440 resolution.
 * Terraria's health/mana style set to 'Fancy 2' in the settings.

**Installation**
 1. Clone this repository.
 2. Install the required dependencies: `pip install -r requirements.txt`

**Running The Agent**
To start the training process, run the main training script: `python rl/train.py`  

This script will launch the environment and begin the training process.
