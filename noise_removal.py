import cv2
import matplotlib.pyplot as plt

def denoise_image(image_path):
    noisy_image = cv2.imread(image_path)
    noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

    gaussian_denoised_img = cv2.GaussianBlur(noisy_image, (5, 5), 0)
    gaussian_denoised_img_rgb = cv2.cvtColor(gaussian_denoised_img, cv2.COLOR_BGR2RGB)

    median_denoised_img = cv2.medianBlur(noisy_image, 5)
    median_denoised_img_rgb = cv2.cvtColor(median_denoised_img, cv2.COLOR_BGR2RGB)

    bilateral_denoised_img = cv2.bilateralFilter(noisy_image, 9, 75, 75)
    bilateral_denoised_img_rgb = cv2.cvtColor(bilateral_denoised_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,8))
    plt.subplot(2, 2, 1)
    plt.title('Noisy Image')
    plt.imshow(noisy_image_rgb)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Gaussian Blur Image')
    plt.imshow(gaussian_denoised_img_rgb)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Median Blur Image')
    plt.imshow(median_denoised_img_rgb)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Bilateral Filter Image')
    plt.imshow(bilateral_denoised_img_rgb)
    plt.axis('off')

    plt.show()

denoise_image('noisy_image1.png')
denoise_image('noisy_image2.jpg')
denoise_image('noisy_image3.webp')