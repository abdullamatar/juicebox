# import torch
from anomalib.data.utils import read_image
from anomalib.deploy.inferencers import TorchInferencer
import os


def infer(img_dir_path=None, model_path=None) -> list:
    """
    Returns a list of predictions for each image in the img_dir_path, optionally probide model_path.
    """
    if img_dir_path is None:
        img_dir_path = "./Reels/heldoutAnomalous"
    if model_path is None:
        model_path = "./weights/torch/model.pt"
    inferencer = TorchInferencer(model_path)
    imgs = []
    predictions = []
    for root, _, files in os.walk(img_dir_path):
        for name in files:
            # print(root)
            # print(os.path.join(root, name))
            imgs.append(read_image(os.path.join(root, name)))
            # print(img)
    for img in imgs:
        predictions.append(inferencer.predict(img))
        # print(inferencer.infer(img))
    # print(next(os.walk("./Reels/heldoutAnomalous")))
    return predictions


# if __name__ == "__main__":
#     print(infer())
