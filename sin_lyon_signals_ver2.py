import numpy as np
import wave
import matplotlib.pyplot as plt
from lyon.calc import LyonCalc

# define sampling rate and time
fs = 16000  
all_time = 5.0  
t = np.arange(0, all_time/4, 1/fs)  

# define signal
def generate_signal(freq):
    return np.sin(2 * np.pi * freq * t)

# generate signalsS
signals = [generate_signal(freq) for freq in [100, 500, 1000, 5000]]
#signals = [generate_signal(freq) for freq in [1000, 5000, 10000, 20000]]

# concatenate signals
full_signal = np.concatenate(signals)

# Convert the signal to 16-bit data.
signal_to_save = (full_signal * 32767).astype(np.int16)

# Write the signal to a wave file
with wave.open('full_signal.wav', 'w') as wave_file:
    wave_file.setnchannels(1)  # mono
    wave_file.setsampwidth(2)  # 2 bytes for int16
    wave_file.setframerate(fs)
    wave_file.writeframes(signal_to_save.tobytes())

print(f'full_signal.shape: {full_signal.shape}')
print(f'full signal length: {len(full_signal)/fs} sec')

calc = LyonCalc()
coch = calc.lyon_passive_ear(full_signal, fs, 8)
print(coch.shape)

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
im = axes.imshow(coch.T, 
            aspect='auto', 
            origin='lower', 
            cmap='jet',
            extent=[0, len(full_signal)/fs, 0, 86]
            )
axes.set_title(f'Chochleagram')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Frequency (channels)')
fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('cochleagram.png')