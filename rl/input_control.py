import pyautogui
import pygetwindow as gw
import time

windows = gw.getWindowsWithTitle('Terraria')

if windows:
    print("Terraria window found.")
    terraria_window = windows[0]
    terraria_window.activate()
    time.sleep(2)  # Time buffer for me to display window
    
    # Hold the key down for movement
    print("Moving right...")
    pyautogui.keyDown('d')
    time.sleep(1)  # Hold for 1 second
    pyautogui.keyUp('d')
    
    time.sleep(0.5)
    
    print("Jumping")
    pyautogui.keyDown('space')
    time.sleep(0.3)
    pyautogui.keyUp('space')
else:
    print("Terraria window not found!")