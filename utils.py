import os

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import Module
from torchvision.utils import make_grid, save_image


def get_noise(n_samples: int, z_dim: int, device: str = "cuda"):
    """
    Function for returning random noise from a normal distribution with mean 0 and variance 1.
    """
    return torch.randn(n_samples, z_dim, 1, 1, device=device)


def save_tensor_images(image_tensor: Tensor, save_path: str, num_images: int = 25, nrow: int = 5) -> None:
    """
    Function for saving images in an uniform grid.
    """
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow, normalize=True)
    save_image(image_grid, save_path)


def weights_init(m: Module) -> None:
    """
    Function for initializing weights.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def make_gif(imgs_dir: str, save_path: str) -> None:
    """
    Function for creating gif from images.
    """
    imgs = os.listdir(path=imgs_dir)
    imgs.sort(key=lambda f: int(f.split(".")[0]))

    frames = [Image.open(imgs_dir + img) for img in imgs]
    frames[0].save(save_path + "learning.gif", format="GIF", append_images=frames, save_all=True, duration=100, loop=0)
