import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# Load audio
fs, signal_original = wavfile.read("audio.wav")

# Convert stereo → mono
if len(signal_original.shape) > 1:
    signal_original = signal_original[:, 0]

# --- Brutal downsampling ---
facteur = 10
signal_sous_ech = signal_original[::facteur]
fs_sous_ech = fs // facteur

wavfile.write("audio_aliasing.wav", fs_sous_ech, signal_sous_ech)

# --- Spectrogram comparison ---
plt.figure(figsize=(12, 5))

# Original
plt.subplot(1, 2, 1)
plt.specgram(signal_original, Fs=fs, NFFT=1024, noverlap=512)
plt.title("Spectrogramme Original (44.1 kHz)")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.ylim(0, fs/2)

# Downsampled
plt.subplot(1, 2, 2)
plt.specgram(signal_sous_ech, Fs=fs_sous_ech, NFFT=1024, noverlap=512)
plt.title("Sous-échantillonné (4.41 kHz)")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.ylim(0, fs_sous_ech/2)

plt.tight_layout()
plt.show()