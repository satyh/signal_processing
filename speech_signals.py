import os
import glob

import wave
import librosa

import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def get_label(file_path):
    labels = []
    for directory in glob.glob(file_path):
        labels.append(directory.split('/')[-1])
    return labels

filenames = glob.glob('data/mini_speech_commands/*/*')
print(filenames[0])
label = filenames[0].split('/')[-2]

for file in filenames:
    # if not wav file
    if not file.endswith('.wav'):
        print(file)

# save file list as csv
with open('speech_signals.csv', 'w') as f:
    for file in filenames:
        f.write(f'{file}\n')

with wave.open(filenames[0], 'rb') as wav_file:
    num_channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    num_samples = num_channels * num_frames
    time = num_frames / frame_rate
    compression_type = wav_file.getcomptype()
    compression_name = wav_file.getcompname()

    print("Number of channels:", num_channels)
    print("Sample width:", sample_width)
    print("Frame rate:", frame_rate)
    print("Number of frames:", num_frames)
    print("Number of samples:", num_samples)
    print("Duration in sec:", time)
    print("Compression type:", compression_type)
    print("Compression name:", compression_name)


plt.figure(figsize=(12, 6))
data, sr = librosa.load(filenames[0], sr=None)
t = np.arange(0, len(data))/sr
plt.plot(t, data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(filenames[0])
plt.grid(True)
plt.savefig('speech_signal.png')
# plt.show()
plt.clf()

def compute_stft(signal, window_size):
    hop_size = int(window_size / 2)
    fft_size = window_size
    spec = []
    window = np.hamming(window_size)
    for i in range(0, len(signal) - window_size, hop_size):
        windowed_signal = window * signal[i:i+window_size]
        spectrum = np.fft.fft(windowed_signal)[:int(fft_size/2)]
        spectrum = np.abs(spectrum)
        spec.append(spectrum)
    spec = np.array(spec).T
    return spec

window_size = [int(sr * ms / 1000) for ms in [25, 125, 375]]
print(window_size)
stft_results = [compute_stft(data, size) for size in window_size]
for result in stft_results:
    print(result.shape)
fs = sr

# plot stft
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for i, (ws, spec) in enumerate(zip(window_size, stft_results)):
    im = axes[i].imshow(spec, 
                   aspect='auto', 
                   origin='lower', 
                   cmap='jet', 
                   extent=[0, len(data)/fs, 0, fs/2])
    axes[i].set_title(f'STFT, window size: {ws/fs*1000} ms')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Frequency (Hz)')
    axes[i].set_ylim(0, 3400)
    fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('speech_stft.png')
# plt.show()
plt.clf()

window_size = [128, 256, 512]
print(window_size)
stft_results = [compute_stft(data, size) for size in window_size]
for result in stft_results:
    print(result.shape)
fs = sr

# plot stft
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for i, (ws, spec) in enumerate(zip(window_size, stft_results)):
    im = axes[i].imshow(spec, 
                   aspect='auto', 
                   origin='lower', 
                   cmap='jet', 
                   extent=[0, len(data)/fs, 0, fs/2])
    axes[i].set_title(f'STFT, window size: {ws/fs*1000} ms')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Frequency (Hz)')
    axes[i].set_ylim(0, 3400)
    fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('speech_stft_v2.png')
# plt.show()
plt.clf()

lengths = []
for file in filenames:
    y, sr = librosa.load(file, sr=None)
    #print(librosa.get_duration(y=y, sr=sr))
    lengths.append(librosa.get_duration(y=y, sr=sr))

plt.figure(figsize=(12, 6))
plt.hist(lengths, bins=100)
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.title('Speech Signal Duration')
plt.grid(True)
plt.savefig('speech_signal_duration.png')
# plt.show()

def resize_image(image, target_height=32, target_width=32):
    original_height, original_width = image.shape
    height_ratio = target_height / original_height
    width_ratio = target_width / original_width
    return zoom(image, (height_ratio, width_ratio))

for result in stft_results:
    h, w = result.shape
    h = int(h * 3400/fs/2)
    result = result[0:h, :]
    print(result.shape)

# plot stft
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for i, (ws, spec) in enumerate(zip(window_size, stft_results)):
    im = axes[i].imshow(resize_image(spec), 
                   aspect='auto', 
                   origin='lower', 
                   cmap='jet', 
                   extent=[0, len(data)/fs, 0, fs/2])
    axes[i].set_title(f'STFT, window size: {ws/fs*1000} ms')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Frequency (Hz)')
    axes[i].set_ylim(0, 3400)
    fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('speech_stft_resize_v2.png')
# plt.show()
plt.clf()