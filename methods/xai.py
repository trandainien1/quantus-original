import captum
import torch
import torchvision

from captum.attr import NoiseTunnel
from torchvision.transforms import functional as Ft

print('[IMPORT]')
from .tam_wrapper import TAMWrapper
from .chefer1_wrapper import Chefer1Wrapper
from .chefer2_wrapper import Chefer2Wrapper
from .rise_wrapper import RISEWrapper
from .rollout_wrapper import RolloutWrapper
from .bt_wrapper import BTTWrapper, BTHWrapper
from .tis_wrapper import TISWrapper
from .cam_wrapper import CAMWrapper
from .inputgrad_wrapper import InputGradWrapper
from .integratedgrad_wrapper import IntegratedGradWrapper
from .lime_wrapper import LimeWrapper
from .occlusion_wrapper import OcclusionWrapper
from .smoothgrad_wrapper import SmoothGradWrapper

print(end='\n')




class Random:
    """
    Output a random map with the same size as the input but 1 channel
    """
    def __init__(self, model, **kwargs):
        pass

    def attribute(self, inputs, target=None):
        return torch.rand(1, 1, *inputs.shape[-2:])

class Sobel:
    """
    Border detection using a Sobel filter
    """
    def __init__(self, model, **kwargs):
        pass

    def attribute(self, inputs, target=None):
        grayscale = Ft.rgb_to_grayscale(inputs)
        sobel_filter = torch.tensor(
            [[[[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]], ],
             [[[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]], ],
            device=inputs.device,
            dtype=inputs.dtype
        )
        x = torch.nn.functional.conv2d(grayscale, sobel_filter)
        x = x.square()
        x = x.sum(1, keepdim=True)
        x = x.sqrt()
        return x

class Gaussian:
    """
    Centered Gaussian 2D
    """
    def __init__(self, model, **kwargs):
        pass

    def attribute(self, inputs, target=None):
        # Get input sizes
        x_size, y_size = inputs.shape[-2:]

        # create a 2d meshgrid from -1 to 1 with same size as input
        x, y = torch.meshgrid(torch.linspace(-1, 1, x_size, device=inputs.device),
                              torch.linspace(-1, 1, y_size, device=inputs.device))

        # distance from the mean (center)
        distance = torch.sqrt(x * x + y * y)

        # Set sigma to 1 (mu = 0)
        sigma = 1

        # Calculating Gaussian array
        gaussian_2d = torch.exp(- (distance ** 2 / (2.0 * sigma ** 2)))

        return gaussian_2d.view(1, 1, x_size, y_size)


### Parameters for each method ###
methods_dict = {
    'lime': {
        'class_fn': LimeWrapper,
    },
    'occlusion': {
        'class_fn': OcclusionWrapper,
    },
    'rise': {
        'class_fn': RISEWrapper,
        'n_masks': 4000,
        'input_size': 224,
        'batch_size': 128,
    },
    'gradcam': {
        'class_fn': CAMWrapper,
    },
    'scorecam': {
        'class_fn':CAMWrapper,
    },
    'gradcam++': {
        'class_fn': CAMWrapper,
    },
    'random': {
        'class_fn': Random,
    },
    'sobel': {
        'class_fn': Sobel,
    },
    'gaussian': {
        'class_fn': Gaussian,
    },
    'inputgrad': {
        'class_fn': InputGradWrapper,
    },
    'integratedgrad': {
        'class_fn': IntegratedGradWrapper,
    },
    'smoothgrad': {
        'class_fn': SmoothGradWrapper,
    },
    'rollout': {
        'class_fn': RolloutWrapper,
        'discard_ratio': 0.9,
        'head_fusion': 'mean',
    },
    'btt': {
        'class_fn': BTTWrapper,
    },
    'bth': {
        'class_fn': BTHWrapper,
    },
    'chefer1': {
        'class_fn': Chefer1Wrapper,
    },
    'chefer2': {
        'class_fn': Chefer2Wrapper,
    },
    'tam': {
        'class_fn': TAMWrapper,
        'start_layer': 0,
        'steps': 20,
    },
    'tis': {
        'class_fn': TISWrapper,
        'n_masks': 1024, 
        'batch_size': 128, 
        'tokens_ratio': 0.5
    }
}

def get_method(name, model, batch_size=16, dataset_name=None):
    cur_dict = methods_dict[name]
    return cur_dict["class_fn"](model, 
                                method_name=name, 
                                batch_size=batch_size, 
                                dataset_name=dataset_name)
