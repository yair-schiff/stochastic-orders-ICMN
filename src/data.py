import numpy as np
import torch

from sklearn.datasets import make_swiss_roll
from torch.distributions import Categorical
from PIL import Image


class CircleOfGaussians:
    """
    Code for this class is largely based on CP-Flow library:
    https://github.com/CW-Huang/CP-Flow/blob/main/data/toy_data.py
    """
    def __init__(self, n_gaussians=8, std=0.02, radius=2.0):
        super(CircleOfGaussians).__init__()
        thetas = torch.arange(0, 2 * np.pi, 2 * np.pi / n_gaussians, out=torch.FloatTensor())
        self.n_mixture = n_gaussians
        self.centers = torch.stack([radius * torch.sin(thetas), radius * torch.cos(thetas)], dim=1)
        self.std = std

    def sample(self, batch_size):
        ks = Categorical(torch.FloatTensor(1, self.n_mixture).fill_(1.0)).sample([batch_size]).squeeze(1)
        return torch.FloatTensor(batch_size, 2).normal_() * self.std + self.centers[ks]


class SwissRoll:
    """
    Code for this class is taken from CP-Flow library:
    https://github.com/CW-Huang/CP-Flow/blob/main/data/toy_data.py
    Swiss roll distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """
    def __init__(self, noise=0.5):
        super(SwissRoll).__init__()
        self.noise = noise

    def sample(self, batch_size):
        return torch.FloatTensor(make_swiss_roll(n_samples=batch_size, noise=self.noise)[0][:, [0, 2]] / 5.)


class ImagePointCloud:
    """
    Code for converting image to point cloud distribution taken from CP-Flow library:
    https://github.com/CW-Huang/CP-Flow/blob/main/train_toy.py
    """
    def __init__(self, image_name, image_path):
        super(ImagePointCloud).__init__()
        self.image_name = image_name
        img = np.array(Image.open(image_path).convert('L'))
        # Flip image on horizontal axis so that it plots correctly during training/validation plotting
        img = np.flipud(img)
        h, w = img.shape
        xx = np.linspace(-4, 4, w)
        yy = np.linspace(-4, 4, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)

        self.means = np.concatenate([xx, yy], 1)
        img = img.max() - img
        self.probs = img.reshape(-1) / img.sum()
        self.std = np.array([8 / w / 2, 8 / h / 2])

    def sample(self, batch_size):
        idxs = np.random.choice(int(self.probs.shape[0]), int(batch_size), p=self.probs)
        m = self.means[idxs]
        samples = np.random.randn(*m.shape) * self.std + m
        return torch.FloatTensor(samples)
