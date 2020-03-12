import numpy as np


def disp_to_depth(disps, original_width=1242, cap=50):
    depths = np.copy(disps)
    depths = np.array(list(map(lambda x: 0.54 * 721 / (original_width * x), depths)))
    depths[depths > cap] = cap
    return depths


def add_gaussian_noise(depthmaps, mu=0, sigma=0.1):
    c, h, w = depthmaps.shape
    gaussian_noise = np.random.normal(mu, sigma, h * w * c).reshape((c, h, w))
    depthmaps = depthmaps + gaussian_noise
    return depthmaps
