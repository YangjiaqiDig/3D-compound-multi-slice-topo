import torch

def max_outputs(outputs_1, outputs_2, outputs_3):
    # outputs_i with prediction probability map, (2, 2, 1250, 1250) -> (batch, classes, dim, dim)
    # max select from classes logits
    probability_map = torch.max(torch.max(outputs_1, outputs_2), outputs_3)

    return probability_map


def mean_outputs(outputs_1, outputs_2, outputs_3):
    print(outputs_1.shape, outputs_2.shape, outputs_3.shape)
    # probability_map =
    return probability_map

# def smooth_gaussian


if __name__ == "__main__":
    # A full forward pass
    im1 = torch.randn(2, 2, 1, 1)
    im2 = torch.randn(2, 2, 1, 1)
    im3 = torch.randn(2, 2, 1, 1)
    prob = torch.max(torch.max(im1, im2), im3)

    print(im1)
    print(im2)
    print(im3)
    print(prob)