from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

class ImageDifference:
    def __init__(self, imageA, imageB):
        self.imageA = cv2.imread(imageA, cv2.IMREAD_GRAYSCALE)
        self.imageB = cv2.imread(imageB, cv2.IMREAD_GRAYSCALE)

    def get_ssim(self):
        return ssim(self.imageA, self.imageB)

    def get_difference_image(self):
        _, diff = cv2.threshold(cv2.absdiff(self.imageA, self.imageB), 25, 255, cv2.THRESH_BINARY)
        return diff

    def save_difference_image(self, output_path):
        diff = self.get_difference_image()
        cv2.imwrite(output_path, diff)

    def are_images_similar(self, threshold=0.99):
        return self.get_ssim() >= threshold

if __name__ == "__main__":
    img_diff = ImageDifference('path_to_first_image.jpg', 'path_to_second_image.jpg')
    
    print(f"SSIM: {img_diff.get_ssim()}")
    if not img_diff.are_images_similar():
        print("Images are different!")
        img_diff.save_difference_image('difference.jpg')
    else:
        print("Images are similar!")