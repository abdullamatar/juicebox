'''
Task 1. Section Extraction
'''

import cv2
import numpy as np
from skimage.feature import match_template
import matplotlib.pyplot as plt
from copy import deepcopy
from imutils.object_detection import non_max_suppression # pip install imutils
from PIL import Image

def generate_sections(reference_image,multi_up_image, output_img_path,threshold=0.8,saveit=False):
    '''
    `final_bbs` is the main output, others are for debugging.
    '''

    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    multi_up_gray = cv2.cvtColor(multi_up_image, cv2.COLOR_BGR2RGB)
    mask = np.zeros(reference_image.shape[:2], dtype=np.uint8)
    mask[:,:mask.shape[1]//2] = 255
    result = cv2.matchTemplate(multi_up_gray, reference_gray, cv2.TM_CCOEFF_NORMED, None,mask)
    print(f"{result.shape} total results found...")

    # threshold = 0.75  # Adjust this threshold as needed
    loc = np.where(result >= threshold)

    # Perform non-maximum suppression.
    # template_h, template_w = reference_gray.shape[:2]
    # rects = []
    # for (x, y) in zip(loc[0], loc[1]):
    #     rects.append((x, y, x + template_w, y + template_h))
    # pick = non_max_suppression(np.array(rects))

    # Define a function to remove duplicates within a specified distance (tolerance)
    def remove_duplicates(points, tolerance, horizontal_tolerance):
        unique_points = []
        unique_indices = []
        for i, (x, y) in enumerate(points):
            is_unique = True
            for j, (x_other, y_other) in enumerate(unique_points):
                if abs(x - x_other) < horizontal_tolerance:
                    is_unique = False
                    break
                distance = np.sqrt((x - x_other) ** 2 + (y - y_other) ** 2)
                if distance <= tolerance:
                    is_unique = False
                    break
            if is_unique:
                unique_points.append((x, y))
                unique_indices.append(i)
        
        return np.array(unique_points), np.array(unique_indices)

    x, y = loc
    print(f"{len(x)} total occurrences found...")

    tolerance = 10  # Adjust this value as needed
    horizontal_tolerance = reference_image.shape[0]//2
    unique_loc, unique_indices = remove_duplicates(np.column_stack((x, y)), tolerance, horizontal_tolerance)
    print(f"{len(unique_indices)} unique found...")

    matched_sections = []
    bb_img = deepcopy(multi_up_image)
    for pt in unique_loc:
        tl=(0,pt[0])
        br=(multi_up_image.shape[1], pt[0] + reference_image.shape[0])
        bb_img = cv2.rectangle(bb_img, tl, br, (0, 0, 255), 8)
        matched_sections.append(multi_up_image[tl[1]:br[1],tl[0]:br[0]] )

    if (saveit):
        for i, matched_img in enumerate(matched_sections):
            # if matched_img.size != 0:
            cv2.imwrite(f"{output_img_path}/{i}.jpg", matched_img)

    final_bbs = [(u[0],0,multi_up_image.shape[1],reference_image.shape[0]) for u in unique_loc]
    return final_bbs, matched_sections, bb_img, result, loc, unique_loc 


import numpy as np
from torchvision import transforms
import torch
'''
Task 2. Anomaly Detection
'''
import os
from pathlib import Path
from anomalib.deploy import TorchInferencer

def find_anomalies (image:np.ndarray,patch_dim=(166,166),saveit=False):
    """
    Splits an image into non-overlapping patches of size (patch_size, patch_size).
    Pads the image with zeros if needed.

    Parameters:
    - `image` : is the section typically
    - patch_dim (int,int): The size of each square patch

    Returns:
    - `abnormal_pathed_coord` is a list of (x,y) of the left top corner of the BB 
    """
    mode_path = "D:/edgehack/juicebox-behind/weights/torch/model_e20.pt"
    inferencer = TorchInferencer(path=mode_path)

    # image_height, image_width, _ = image.shape

    # Define the dimensions of each grid cell
    patch_width, patch_height = patch_dim

    # Calculate the number of rows and columns needed to cover the entire image
    num_rows = (image.shape[0] + patch_height - 1) // patch_height
    num_cols = (image.shape[1] + patch_width - 1) // patch_width


    if saveit:
        # Create a directory to save the cropped images
        output_directory = f'./Patches'
        os.makedirs(output_directory, exist_ok=True)

    pred_array = np.zeros((num_rows, num_cols), dtype=np.float32)

    # Loop through the image and crop it into grids
    for i in range(num_rows):
        for j in range(num_cols):
            # Define the coordinates for cropping
            x1, y1 = j * patch_width, i * patch_height
            x2, y2 = min((j + 1) * patch_width, image.shape[1]), min((i + 1) * patch_height, image.shape[0])

            # Pad the cropped region if it's smaller than 166x166
            cropped_image = image[y1:y2, x1:x2]
            if cropped_image.shape[0] < patch_height or cropped_image.shape[1] < patch_width:
                padded_image = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)
                padded_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
                cropped_image = padded_image
            
            
            pred = inferencer.predict(image=cropped_image).pred_score
            
            pred_array[i, j] = pred

            if saveit:
                # Define the filename for the cropped image
                filename = os.path.join(output_directory, f'{id}_{i}-{j}.bmp')
                # Save the cropped image
                cv2.imwrite(filename, cropped_image)

        print(f"Row: {i}")

    print(f'Done calculating predictions across the grid/section')

    pred_threshold = 0.7
    pred_array_filtered = np.zeros(pred_array.shape)
    pred_array_filtered[pred_array >= pred_threshold] = 1

    abnormal_pathed_indices = np.argwhere(pred_array_filtered == 1) # those that are 1 (predicted abnormal)
    abnormal_pathed_coord = abnormal_pathed_indices * patch_width # indices x (166,166) to get the real BB
    
    return abnormal_pathed_coord

def split_into_patches(image, patch_size=256) -> np.ndarray:
    """
    Splits an image into non-overlapping patches of size (patch_size, patch_size).
    Pads the image with zeros if needed.

    Parameters:
    - image (ndarray): The image to be split. Shape (H, W, C)
    - patch_size (int): The size of each square patch

    Returns:
    - patches (ndarray): The patches. Shape (N, patch_size, patch_size, C)
    """
    # Image dimensions
    H, W, C = image.shape

    # Calculate required padding
    pad_H = (patch_size - H % patch_size) % patch_size
    pad_W = (patch_size - W % patch_size) % patch_size

    # Pad the image with zeros
    padded_image = np.pad(image, ((0, pad_H), (0, pad_W), (0, 0)), "constant")

    # Reshape into patches
    reshaped_image = padded_image.reshape(
        padded_image.shape[0] // patch_size,
        patch_size,
        padded_image.shape[1] // patch_size,
        patch_size,
        C,
    )

    # Transpose and reshape to finalize the patches
    patches = reshaped_image.transpose(0, 2, 1, 3, 4).reshape(
        -1, patch_size, patch_size, C
    )

    return patches


def normalize_patches(patches):
    """
    Normalize image patches using ImageNet norms.
    Returns:
    - normalized_patches (ndarray): The normalized patches. Shape (N, H, W, C)
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # Convert to PyTorch tensor and change to (N, C, H, W)
    # ? RM'd float conversion
    # torch_tensor = torch.from_numpy(patches)
    patches_tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).float()

    # Apply normalization
    normalized_patches_tensor = normalize(patches_tensor)

    # Convert back to numpy and change to (N, H, W, C)
    normalized_patches = normalized_patches_tensor.permute(0, 2, 3, 1).cpu().numpy()

    return np.asarray(normalized_patches, dtype=np.uint8)