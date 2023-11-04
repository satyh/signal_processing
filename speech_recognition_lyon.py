import os

# os.system('wget http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip')
# os.system('unzip mini_speech_commands.zip')

# must install lyon
# os.system('pip install lyon')

import glob
from tqdm import tqdm

import librosa
from lyon.calc import LyonCalc

import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from sklearn.preprocessing import LabelBinarizer

def get_label(file_path):
    labels = []
    for directory in glob.glob(file_path):
        labels.append(directory.split('/')[-1])
    return labels

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
    dataset = np.array(dataset, np.float64)
    labels = np.array(labels)
    return dataset, labels

filenames = glob.glob('mini_speech_commands/*/*')

dataset, labels = get_audio(filenames)
print(dataset.shape)
print(labels.shape)

unique_labels = np.unique(labels)
print("Unique labels: ", unique_labels)

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
print("One hot labels shape: ", labels.shape)

def normalize(data, mean, variance):
    return (data - mean) / np.sqrt(variance + 1e-5)

def resize_image(image, target_height=32, target_width=32):
    original_height, original_width, channels = image.shape
    height_ratio = target_height / original_height
    width_ratio = target_width / original_width
    return zoom(image, (height_ratio, width_ratio, 1))

calc = LyonCalc()
def get_lyon(data):
    dataset = []
    for i in tqdm(range(len(data))):
        data_lyon = calc.lyon_passive_ear(data[i], 16000, 8)
        dataset.append(data_lyon)
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

def get_resize(dataset, target_height=32, target_width=32):
    b, h, w = dataset.shape
    dataset = dataset.reshape(b, h, w, 1)
    resized_ds = np.array([resize_image(batch, target_height, target_width) for batch in dataset])
    return resized_ds

dataset = get_lyon(dataset)
print(dataset.shape)
dataset = get_normalize(dataset)
print(dataset.shape)

dataset = get_resize(dataset)
print(dataset.shape)

indices = np.arange(len(dataset))
np.random.shuffle(indices)

dataset = dataset[indices]
labels = labels[indices]

split_index = int(len(dataset) * 0.8)
train_dataset = dataset[:split_index]
train_labels = labels[:split_index]
test_dataset = dataset[split_index:]
test_labels = labels[split_index:]

train_dataset = train_dataset.reshape(-1, 32, 32, 1)
test_dataset = test_dataset.reshape(-1, 32, 32, 1)

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = get_model()
model.summary()

model.fit(train_dataset, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_dataset, test_labels, verbose=2)

print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)