import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

def quantifier_image(image, niveaux):
    img_norm = image / 255.0
    img_quant = np.round(img_norm * (niveaux - 1)) / (niveaux - 1)
    return (img_quant * 255).astype(np.uint8)

# Quantization
img_2bits = quantifier_image(img, 4)
img_1bit = quantifier_image(img, 2)

# --- Pixelisation ---
h, w = img.shape
img_small = cv2.resize(img, (w//8, h//8), interpolation=cv2.INTER_NEAREST)
img_pixel = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_NEAREST)

# --- Display ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Originale (8 bits)")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(img_2bits, cmap='gray')
plt.title("2 bits (4 niveaux)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(img_1bit, cmap='gray')
plt.title("1 bit (2 niveaux)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(img_pixel, cmap='gray')
plt.title("Pixelisée (÷8)")
plt.axis("off")

plt.tight_layout()
plt.show()