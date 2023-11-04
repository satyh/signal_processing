import glob
import librosa

import numpy as np

from lyon.calc import LyonCalc
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


filenames = glob.glob('data/mini_speech_commands/*/*')
print(filenames[0])
data, sr = librosa.load(filenames[0], sr=None)
data = np.array(data, np.float64)

calc = LyonCalc()
coch = calc.lyon_passive_ear(data, sr, 8)
print(coch.shape)

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
im = axes.imshow(coch.T, 
            aspect='auto', 
            origin='lower', 
            cmap='jet',
            extent=[0, len(data)/sr, sr/2, 0]
            )
axes.set_title(f'Chochleagram')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Frequency (Hz)')
fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('speech_cochleagram.png')

def resize_image(image, target_height=32, target_width=32):
    original_height, original_width = image.shape
    height_ratio = target_height / original_height
    width_ratio = target_width / original_width
    return zoom(image, (height_ratio, width_ratio))

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
im = axes.imshow(resize_image(coch.T), 
            aspect='auto', 
            origin='lower', 
            cmap='jet',
            extent=[0, len(data)/sr, sr/2, 0]
            )
axes.set_title(f'Chochleagram')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Frequency (Hz)')
fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('speech_cochleagram_resize.png')