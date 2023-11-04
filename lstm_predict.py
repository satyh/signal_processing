import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)

n_steps = 50
batch_size = 10000

series = generate_time_series(batch_size, n_steps + 1)
print(series.shape)

plt.figure(figsize=(11, 4))
plt.plot(series[0, :, 0])
plt.savefig('series.png')
plt.show()
plt.clf()

X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam')
model.summary()

history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('loss.png')
plt.show()
plt.clf()

y_pred = model.predict(X_test)

plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.savefig('pred.png')
plt.show()
plt.clf()