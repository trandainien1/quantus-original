import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import cm

try:
    from .grad_cam.pytorch_grad_cam.grad_cam import GradCAM
    from .grad_cam.pytorch_grad_cam.score_cam import ScoreCAM
    from .grad_cam.pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
    print('SUCCESS: cam was successfully imported.')
except:
    print('ERROR: cam was not found.')

# methods = \
#     {"gradcam": GradCAM,
#     "scorecam": ScoreCAM,
#     "gradcam++": GradCAMPlusPlus}

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class CAMWrapper:
    def __init__(self, model, method_name='gradcam', aug_smooth=False, eigen_smooth=False, **kwargs):
        self.model = model
        self.method_name = method_name
        self.target_layers = [self.model.blocks[-1].norm1]
        self.aug_smooth = aug_smooth
        self.eigen_smooth = eigen_smooth

    def attribute(self, x, target=None):
        cam = methods[self.method_name](model=self.model,
                                target_layers=self.target_layers,
                                use_cuda=True,
                                reshape_transform=reshape_transform)

        grayscale_cam = cam(input_tensor=x,
                            eigen_smooth=self.eigen_smooth,
                            aug_smooth=self.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        return grayscale_cam
