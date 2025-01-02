import numpy as np
import cv2

def render_pendulum_image(theta, image_size=64):
    # Create a blank image
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    # Define the pendulum's parameters
    center = (image_size // 2, image_size // 2)  
    length = image_size // 2 - 4  
    thickness = 3  

    # Compute the pendulum's end-point
    end_x = int(center[0] + length * np.sin(theta))
    end_y = int(center[1] - length * np.cos(theta))  

    # Draw the pendulum
    cv2.line(image, center, (end_x, end_y), color=255, thickness=thickness)

    # Normalize to [0, 1] and return
    return image / 255.0
