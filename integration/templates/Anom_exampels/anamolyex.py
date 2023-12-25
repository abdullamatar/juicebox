from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np


class ImageDifference:
    def _init_(self, imageA_path, imageB_path):
        self.imageA = cv2.imread(imageA_path, cv2.IMREAD_GRAYSCALE)
        self.imageB = cv2.imread(imageB_path, cv2.IMREAD_GRAYSCALE)
        self.labelsA = self.segment_labels(self.imageA)

    def segment_labels(self, image):
        # Threshold the image
        _, thresh = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(contour) for contour in contours]

    def compare_labels_with_imageB(self):
        for label in self.labelsA:
            x, y, w, h = label
            segmented_label = cv2.resize(self.imageA[y:y + h, x:x + w], (self.imageB.shape[1], self.imageB.shape[0]))
            similarity = ssim(segmented_label, self.imageB)
            print(f"SSIM for label at position {x, y}: {similarity}")

if _name_ == "_main_":
    img_diff = ImageDifference('main.jpg', 'multiple.jpg')
    img_diff.compare_labels_with_imageB()