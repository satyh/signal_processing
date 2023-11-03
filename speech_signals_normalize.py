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

def normalize(data, mean, variance):
    return (data - mean) / np.sqrt(variance + 1e-5)

def resize_image(image, target_height=32, target_width=32):
    original_height, original_width, channels = image.shape
    height_ratio = target_height / original_height
    width_ratio = target_width / original_width
    return zoom(image, (height_ratio, width_ratio, 1))

def pat_audio(data):
    if len(data) < 16000:
        data = np.pad(data, (0, 16000 - len(data)), 'constant')
    else:
        data = data[:16000]
    return data

def get_audio(filenames):
    dataset = []
    labels = []
    for i in range(len(filenames)):
        data, _ = librosa.load(filenames[i], sr=None)
        label = filenames[i].split('/')[-2]
        data = pat_audio(data)
        # data = compute_stft(data, window_size)
        # print(data.shape)
        dataset.append(data)
        labels.append(label)
    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels

def get_spectrogram(datas, window_size):
    dataset = []
    for data in datas:
        data = compute_stft(data, window_size)
        dataset.append(data)
    dataset = np.array(dataset)
    return dataset

def get_normalize(dataset):
    means = []
    variances = []
    for batch in dataset:
        means.append(np.mean(batch, axis=(0, 1)))
        variances.append(np.var(batch, axis=(0, 1)))
    mean = np.mean(means, axis=0)
    variance = np.mean(variances, axis=0)
    normalized_ds = np.array([normalize(batch, mean, variance) for batch in dataset])
    return normalized_ds

def resize_image(image, target_height=32, target_width=32):
    original_height, original_width = image.shape
    height_ratio = target_height / original_height
    width_ratio = target_width / original_width
    return zoom(image, (height_ratio, width_ratio))

def get_resize(dataset, target_height=32, target_width=32):
    resized_ds = np.array([resize_image(batch, target_height, target_width) for batch in dataset])
    return resized_ds

if __name__ == "__main__":
    filenames = glob.glob('data/mini_speech_commands/*/*')

    dataset, labels = get_audio(filenames)
    print(dataset.shape)
    print(labels.shape)

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

    dataset = get_spectrogram(dataset, 256)
    print(dataset.shape)

    
    dataset = get_normalize(dataset)
    print(dataset.shape)
    dataset = get_resize(dataset)
    print(dataset.shape)
    

    # np.random.shuffle(dataset)

    # print(dataset.shape)
    # dataset = get_normalize(dataset)
    # print(dataset.shape)
    # dataset = get_resize(dataset)
    # print(dataset.shape)
