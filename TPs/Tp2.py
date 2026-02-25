import numpy as np
import matplotlib.pyplot as plt


#Analyse spectrale

fs = 44100
duration = 1
t = np.arange(0, duration, 1/fs)

f1 = 440
f2 = 880
signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# Compute FFT
fft_signal = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), 1/fs)

plt.figure(figsize=(10, 5))
plt.plot(freqs[:len(freqs)//2], np.abs(fft_signal)[:len(freqs)//2])
plt.xlim(0, 2000)
plt.title("Spectre du signal (Pics à 440 Hz et 880 Hz)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


# Ajout de bruit

f_bruit = 5000
bruit = 0.5 * np.sin(2 * np.pi * f_bruit * t)
signal_bruite = signal + bruit

fft_bruite = np.fft.fft(signal_bruite)

plt.figure(figsize=(10, 5))
plt.plot(freqs[:len(freqs)//2], np.abs(fft_bruite)[:len(freqs)//2])
plt.xlim(0, 6000)
plt.title("Spectre du signal bruité (Pic à 5000 Hz visible)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


# Filtrage Coupe-Bande

mask = (np.abs(freqs) > 4900) & (np.abs(freqs) < 5100)
fft_bruite[mask] = 0

# Inverse FFT
signal_filtre = np.real(np.fft.ifft(fft_bruite))


# Comparaison Temporelle

plt.figure(figsize=(10, 5))
plt.plot(t[:2000], signal[:2000], label="Original")
plt.plot(t[:2000], signal_filtre[:2000], label="Filtré", alpha=0.7)
plt.title("Comparaison Signal Original vs Signal Filtré")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()