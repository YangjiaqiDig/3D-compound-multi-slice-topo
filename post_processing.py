import torch
from torch import nn
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import glob
import numpy as np
import math

def max_outputs(outputs_1, outputs_2, outputs_3):
    # outputs_i with prediction probability map, (2, 2, 1250, 1250) -> (batch, classes, dim, dim)
    # max select from classes logits
    probability_map = torch.max(torch.max(outputs_1, outputs_2), outputs_3)

    return probability_map


def mean_outputs(outputs_1, outputs_2, outputs_3):
    # print(outputs_1.shape, outputs_2.shape, outputs_3.shape)
    probability_map = (outputs_1 + outputs_2 + outputs_3) / 3

    return probability_map

def smooth_gaussian(probability_map):
    kernel_size = 6
    sigma = 1
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(2, 1, 1, 1)
    print(gaussian_kernel, gaussian_kernel.shape)


    gaussian_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                kernel_size=kernel_size, groups=1, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    probability_map = gaussian_filter
    return gaussian_filter


if __name__ == "__main__":
    # A full forward pass
    im1 = torch.randn(2, 2, 1, 1)
    im2 = torch.randn(2, 2, 1, 1)
    im3 = torch.randn(2, 2, 1, 1)
    prob = torch.max(torch.max(im1, im2), im3)

    print(im1)
    print(im2)
    print(im3)
    print(im1+im2)


    image_name = glob.glob("train_wrongDt/label/1.png")
    image = Image.open(image_name[0])
    image.show()
    img_as_np = np.asarray(image)
    img_as_np = smooth_gaussian(torch.Tensor(img_as_np))
    img1 = Image.fromarray(img_as_np)
    img1.show()