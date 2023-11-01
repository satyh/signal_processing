import numpy as np
import matplotlib.pyplot as plt

# サンプリング周波数を定義
fs = 16000  # サンプリング周波数 16000Hz
all_time = 2.0  # 20秒間の時間ベクトル
t = np.arange(0, all_time/4, 1/fs)  # 5秒間の時間ベクトル

# 各周波数での信号を生成する関数
def generate_signal(freq):
    return np.sin(2 * np.pi * freq * t)

# 各周波数での信号を生成
signals = [generate_signal(freq) for freq in [10, 25, 50, 100]]

# 信号を連結
full_signal = np.concatenate(signals)

print(f'full_signal.shape: {full_signal.shape}')
print(f'full signal length: {len(full_signal)/fs} sec')

# 連結した信号をプロット
full_time = np.arange(0, all_time, 1/fs)  # 20秒間の時間ベクトル
plt.figure(figsize=(18, 6))
plt.plot(full_time, full_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal with Changing Frequencies')
plt.grid(True)
plt.savefig('signal.png')
# plt.show()

# def compute_stft(signal, window_size):
#     hop_size = int(window_size / 2)
#     fft_size = window_size
#     # 窓関数を定義
#     window = np.hamming(window_size)
#     # 短時間フーリエ変換を行う
#     result = np.array([np.fft.fft(window * signal[i:i+window_size], fft_size) for i in range(0, len(signal)-window_size, hop_size)])
#     # 振幅スペクトルを計算
#     mag = np.abs(result).T
#     return mag

def compute_stft(signal, window_size):
    hop_size = int(window_size / 2)
    fft_size = window_size
    spec = []
    # 窓関数を定義
    window = np.hamming(window_size)
    for i in range(0, len(signal) - window_size, hop_size):
        windowed_signal = window * signal[i:i+window_size]
        spectrum = np.fft.fft(windowed_signal)[:int(fft_size/2)]
        spectrum = np.abs(spectrum)
        spec.append(spectrum)
    spec = np.array(spec).T
    return spec

# 窓サイズを定義
window_size = [int(fs * ms / 1000) for ms in [25, 125, 375, 1000]]
print(window_size)
stft_results = [compute_stft(full_signal, size) for size in window_size]
for result in stft_results:
    print(result.shape)

# 短時間フーリエ変換の結果をプロット
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
for i, (ws, spec) in enumerate(zip(window_size, stft_results)):
    # axes[i].imshow(20 * np.log10(spec), 
    #                aspect='auto', 
    #                origin='lower', 
    #                cmap='jet', 
    #                extent=[0, len(full_signal)/fs, 0, fs/2])
    im = axes[i].imshow(spec, 
                   aspect='auto', 
                   origin='lower', 
                   cmap='jet', 
                   extent=[0, len(full_signal)/fs, 0, fs/2])
    axes[i].set_title(f'STFT, window size: {ws/fs*1000} ms')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Frequency (Hz)')
    axes[i].set_ylim(0, 150)
    fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('stft.png')
# plt.show()
# reset plt
plt.clf()

# 短時間フーリエ変換の結果をプロット
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
for i, (ws, spec) in enumerate(zip(window_size, stft_results)):
    log_spec = 20 * np.log10(spec)
    im = axes[i].imshow(log_spec, 
                   aspect='auto', 
                   origin='lower', 
                   cmap='jet', 
                   extent=[0, len(full_signal)/fs, 0, fs/2],
                   vmin=-10,
                   vmax=100)
    axes[i].set_title(f'STFT, window size: {ws/fs*1000} ms')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Frequency (Hz)')
    axes[i].set_ylim(0, 150)
    fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('stft_log.png')
# plt.show()