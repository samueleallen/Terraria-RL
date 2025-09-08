# Specifications

## Feature: 
filter_char_imgs

### Purpose: 
To fix any preprocessed images (of numbers) that messed up during the contour detection stage.

### Assumptions: 
The screenshots are already preprocessed. The string should contain a '/' and the right side should be 3 digits long. Assumes right side was predicted correctly.

### Inputs: 
A list of preprocessed screenshots of the predicted locations of each character

### Outputs: 
A list of preprocessed screenshots of the correct locations of each character

### State Changes: 
The list may change in size if detected to be incorrect

### Cases & Expected Behaviors:
    Case 1 (Easy): 4 contours detected, 4 chars predicted
        * This means we need to split our first char image in half and predict each side as different chars. In this case the right should be a '/' and the left should be an integer. Then return a new list with the new split image
        * There should always be a minimum of 5 characters predicted so this case is guranteed to have an error.
    Case 2 (Hard): 5 contours detected, 5 chars predicted
        * This means we need to split our 2nd char image roughly in half and predict each side as different characters. The right should be a slash, and the left should be a number. 
        * Not guranteed to be an error. There could be accurate predictions like 0/100, 5/100, etc. where there should be 5 chars predicted.
            1. Accurate prediction: We need to predict the 1st and 2nd image with our CNN and see if the first image is an int and the second image is a '/'. If true, return list of images. Else, continue with case 2.2
            2. Inaccurate prediction (1st and 2nd digit): We need to split the 1st image in half and predict the left and right side. The right side should be a '/' and the left should be an integer. If true, return new list of images with first image replaced with two new elements which are the left and right half of first image. Else, continue with case 2.3
            3. Innacurate prediction (2nd and 3rd digit): We need to split the second image in half of the original list and predict the left and right side. The right side should be a '/' and the left should be an integer. If true, return new list of images with second image replaced with two new elements which are the left and right half of second image. Else, we return an arbitrary value like 10. Later, we nede to check for if the function returned 10, if it did, we don't feed the value to our model.

