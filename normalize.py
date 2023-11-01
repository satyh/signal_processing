import numpy as np

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