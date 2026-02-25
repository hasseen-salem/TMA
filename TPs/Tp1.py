import numpy as np
import matplotlib.pyplot as plt

# 1.1 Génération d'un signal sinusoïdal
fs = 100
t = np.arange(0, 1, 1/fs)
x = 1 * np.sin(2 * np.pi * 10 * t)

plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.title("Signal sinusoïdal pur")

# 1.2 Ajout de Bruit (BBG)
bruit = np.random.normal(0, 0.3, len(t))
y = x + bruit

plt.figure(figsize=(10, 4))
plt.plot(t, x, label="Pur")
plt.plot(t, y, label="Bruité", alpha=0.7)
plt.legend()
plt.title("Signal pur vs Bruité")

# 1.3 Porte et Convolution
porte = np.zeros(100)
porte[20:41] = 1  # 1 entre n=20 et n=40

# La convolution de deux portes rectangulaires donne une forme triangulaire
convolution = np.convolve(porte, porte)
plt.figure()
plt.plot(convolution)
plt.title("Convolution de la porte avec elle-même (Forme de Triangle)")
plt.show()