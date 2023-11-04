import numpy as np
import matplotlib.pyplot as plt
from lyon.calc import LyonCalc

# define sampling rate and time
fs = 16000  
all_time = 2.0  
t = np.arange(0, all_time/4, 1/fs)  

# define signal
def generate_signal(freq):
    return np.sin(2 * np.pi * freq * t)

# generate signalsS
signals = [generate_signal(freq) for freq in [10, 25, 50, 100]]

# concatenate signals
full_signal = np.concatenate(signals)

print(f'full_signal.shape: {full_signal.shape}')
print(f'full signal length: {len(full_signal)/fs} sec')

calc = LyonCalc()
coch = calc.lyon_passive_ear(full_signal, fs, 8)
print(coch.shape)

fig, axes = plt.subplots(1, 1, figsize=(10, 8))
im = axes.imshow(coch.T, 
            aspect='auto', 
            origin='lower', 
            cmap='jet'
            #extent=[0, len(full_signal)/fs, 0, fs/2]
            )
axes.set_title(f'Chochleagram')
axes.set_xlabel('Time (s)')
axes.set_ylabel('Frequency (Hz)')
fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('cochleagram.png')

# # plot signal
# full_time = np.arange(0, all_time, 1/fs)  # 20秒間の時間ベクトル
# plt.figure(figsize=(18, 6))
# plt.plot(full_time, full_signal)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Signal with Changing Frequencies')
# plt.grid(True)
# plt.savefig('signal.png')
# # plt.show()
# plt.clf()

# def compute_stft(signal, window_size):
#     hop_size = int(window_size / 2)
#     fft_size = window_size
#     spec = []
#     window = np.hamming(window_size)
#     for i in range(0, len(signal) - window_size, hop_size):
#         windowed_signal = window * signal[i:i+window_size]
#         spectrum = np.fft.fft(windowed_signal)[:int(fft_size/2)]
#         spectrum = np.abs(spectrum)
#         spec.append(spectrum)
#     spec = np.array(spec).T
#     return spec

# # calculate stft
# window_size = [int(fs * ms / 1000) for ms in [25, 125, 375, 1000]]
# print(window_size)
# stft_results = [compute_stft(full_signal, size) for size in window_size]
# for result in stft_results:
#     print(result.shape)

# # plot stft
# fig, axes = plt.subplots(4, 1, figsize=(10, 8))
# for i, (ws, spec) in enumerate(zip(window_size, stft_results)):
#     # axes[i].imshow(20 * np.log10(spec), 
#     #                aspect='auto', 
#     #                origin='lower', 
#     #                cmap='jet', 
#     #                extent=[0, len(full_signal)/fs, 0, fs/2])
#     im = axes[i].imshow(spec, 
#                    aspect='auto', 
#                    origin='lower', 
#                    cmap='jet', 
#                    extent=[0, len(full_signal)/fs, 0, fs/2])
#     axes[i].set_title(f'STFT, window size: {ws/fs*1000} ms')
#     axes[i].set_xlabel('Time (s)')
#     axes[i].set_ylabel('Frequency (Hz)')
#     axes[i].set_ylim(0, 150)
#     fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
# plt.tight_layout()
# plt.savefig('stft.png')
# # plt.show()
# plt.clf()

# # plot stft (log scale)
# fig, axes = plt.subplots(4, 1, figsize=(10, 8))
# for i, (ws, spec) in enumerate(zip(window_size, stft_results)):
#     log_spec = 20 * np.log10(spec)
#     im = axes[i].imshow(log_spec, 
#                    aspect='auto', 
#                    origin='lower', 
#                    cmap='jet', 
#                    extent=[0, len(full_signal)/fs, 0, fs/2],
#                    vmin=-10,
#                    vmax=100)
#     axes[i].set_title(f'STFT, window size: {ws/fs*1000} ms')
#     axes[i].set_xlabel('Time (s)')
#     axes[i].set_ylabel('Frequency (Hz)')
#     axes[i].set_ylim(0, 150)
#     fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
# plt.tight_layout()
# plt.savefig('stft_log.png')
# # plt.show()