import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np

def verify_preprocessing(data_dir):
    # Check pose estimation
    pose_dir = os.path.join(data_dir, 'pose')
    if not os.path.exists(pose_dir):
        print("Pose directory not found!")
        return False
    
    # Check image masks
    image_mask_dir = os.path.join(data_dir, 'image-mask')
    cloth_mask_dir = os.path.join(data_dir, 'cloth-mask')
    if not os.path.exists(image_mask_dir) or not os.path.exists(cloth_mask_dir):
        print("Mask directories not found!")
        return False
    
    # Visual inspection
    sample_image = os.listdir(os.path.join(data_dir, 'images'))[0]
    sample_name = os.path.splitext(sample_image)[0]
    
    image = cv2.imread(os.path.join(data_dir, 'images', sample_image))
    pose = json.load(open(os.path.join(pose_dir, f"{sample_name}_keypoints.json")))
    image_mask = cv2.imread(os.path.join(image_mask_dir, f"{sample_name}.png"), 0)
    cloth_mask = cv2.imread(os.path.join(cloth_mask_dir, f"{sample_name}.png"), 0)
    
    plt.figure(figsize=(20, 5))
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    
    plt.subplot(142)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for keypoint in pose['people'][0]['pose_keypoints_2d']:
        plt.plot(keypoint[0], keypoint[1], 'ro')
    plt.title("Pose Keypoints")
    
    plt.subplot(143)
    plt.imshow(image_mask, cmap='gray')
    plt.title("Image Mask")
    
    plt.subplot(144)
    plt.imshow(cloth_mask, cmap='gray')
    plt.title("Cloth Mask")
    
    plt.show()
    
    return True

# Usage
if __name__ == "__main__":
    data_dir = "path/to/your/data/directory"
    if verify_preprocessing(data_dir):
        print("Preprocessing verified successfully!")
    else:
        print("Preprocessing verification failed!")
