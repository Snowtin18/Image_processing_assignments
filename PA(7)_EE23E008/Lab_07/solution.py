import numpy as np
import cv2
import matplotlib.pyplot as plt

def dft_1d(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    twiddle_factor = np.exp(-2j * np.pi * k * n / N)
    return np.dot(twiddle_factor, signal)

def dft_2d(image):
    #Peforming 2D dft using row-column decompositions
    # Perform 1D DFT along rows
    dft_rows = np.apply_along_axis(dft_1d, 1, image.astype(float))

    # Perform 1D DFT along columns
    dft_columns = np.apply_along_axis(dft_1d, 0, dft_rows.astype(float))

    # Shift zero frequency components to the center
    dft_shifted = np.fft.fftshift(dft_columns)

    magnitude=np.abs(dft_shifted)
    phase=np.angle(dft_shifted)

    return magnitude,phase

def idft_1d(spectrum):
    N = len(spectrum)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return np.dot(e, spectrum) / N

def idft_2d(spectrum):
    # Perform 1D inverse DFT along rows
    idft_rows = np.apply_along_axis(idft_1d, 1, spectrum)
    # Perform 1D inverse DFT along columns
    idft_result = np.apply_along_axis(idft_1d, 0, idft_rows)
    return idft_result



# Read the input image
image1 = cv2.imread('fourier.png', cv2.IMREAD_GRAYSCALE)

image2 = cv2.imread('fourier_transform.png', cv2.IMREAD_GRAYSCALE)

# Calculate magnitude spectrum (logarithmic scale)


magnitude1,phase1=dft_2d(image1)
magnitude2,phase2=dft_2d(image2)

reconstructed_spectrum1 = magnitude2 * np.exp(1j * phase1)
reconstructed_spectrum2 = magnitude1 * np.exp(1j * phase2)

reconstructed_image1 = np.abs(idft_2d(reconstructed_spectrum1))
reconstructed_image2 = np.abs(idft_2d(reconstructed_spectrum2))

# Plot the original and reconstructed images
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Original Image 1')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(image2, cmap='gray')
plt.title('Original Image 2')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(reconstructed_image1, cmap='gray')
plt.title('Reconstructed Image 1')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(reconstructed_image2, cmap='gray')
plt.title('Reconstructed Image 2')
plt.axis('off')

plt.tight_layout()
plt.show()
