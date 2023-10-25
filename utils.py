import numpy as np
from torchvision import transforms
import torch


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
