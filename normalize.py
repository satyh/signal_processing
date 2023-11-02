import numpy as np
from scipy.ndimage import zoom

spectrogram_ds = np.array([np.random.rand(64, 64, 1) for _ in range(512)])
print(spectrogram_ds.shape)

means = []
variances = []

for batch in spectrogram_ds:
    means.append(np.mean(batch, axis=(0, 1)))
    variances.append(np.var(batch, axis=(0, 1)))

mean = np.mean(means, axis=0)
variance = np.mean(variances, axis=0)

print(mean)
print(variance)

def normalize(data, mean, variance):
    return (data - mean) / np.sqrt(variance + 1e-5)

normalized_ds = np.array([normalize(batch, mean, variance) for batch in spectrogram_ds])

def resize_image(image, target_height=32, target_width=32):
    original_height, original_width, channels = image.shape
    height_ratio = target_height / original_height
    width_ratio = target_width / original_width
    return zoom(image, (height_ratio, width_ratio, 1))

image = np.random.rand(64, 64, 1)
print(image.shape)
print(resize_image(image).shape)
