import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse


# ─────────────────────────────
# ADD NOISE (make image dirty)
# ─────────────────────────────

def gaussian_noise(img):
    noise = np.random.normal(0, 25, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def salt_pepper_noise(img):
    out = img.copy()
    h, w = img.shape[:2]

    # random white points (salt)
    for _ in range(3000):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        out[y, x] = 255

    # random black points (pepper)
    for _ in range(3000):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        out[y, x] = 0

    return out


def uniform_noise(img):
    noise = np.random.uniform(-40, 40, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


# ─────────────────────────────
# REMOVE NOISE (clean image)
# ─────────────────────────────

def mean_filter(img):
    return cv2.blur(img, (3, 3))


def gaussian_filter(img):
    return cv2.GaussianBlur(img, (5, 5), 1)


def median_filter(img):
    return cv2.medianBlur(img, 5)


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


# ─────────────────────────────
# QUALITY CHECK
# ─────────────────────────────

def check_quality(original, result):
    return (
        round(psnr(original, result, data_range=255), 2),
        round(mse(original, result), 2)
    )


# ─────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────

def run_demo(image_path=None):

    # STEP 1: LOAD IMAGE
    if image_path:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        print("Using your image")
    else:
        img = create_simple_image()
        print("Using sample image")

    noises = [
        ("Gaussian Noise", gaussian_noise),
        ("Salt & Pepper", salt_pepper_noise),
        ("Uniform Noise", uniform_noise)
    ]

    filters = [
        ("Mean", mean_filter),
        ("Gaussian", gaussian_filter),
        ("Median", median_filter),
        ("Bilateral", bilateral_filter)
    ]

    results = []

    # STEP 2: APPLY NOISE + FILTERS
    for name, noise_fn in noises:
        noisy = noise_fn(img)

        fig, ax = plt.subplots(1, 6, figsize=(14, 4))
        fig.suptitle(name)

        ax[0].imshow(img)
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(noisy)
        ax[1].set_title("Noisy")
        ax[1].axis("off")

        best = ""
        best_score = 0

        # apply filters
        for i, (fname, f) in enumerate(filters):
            result = f(noisy)
            p, m = check_quality(img, result)

            ax[i + 2].imshow(result)
            ax[i + 2].set_title(fname)
            ax[i + 2].axis("off")

            print(f"{name} + {fname} → PSNR={p}, MSE={m}")

            if p > best_score:
                best_score = p
                best = fname

        results.append((name, best, best_score))

        plt.tight_layout()

        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{name.replace(' ', '_')}.png")
        plt.close()

    # STEP 3: FINAL RESULT
    print("\nBEST RESULTS:")
    for n, b, s in results:
        print(f"{n} → Best Filter: {b} (PSNR {s})")


# ─────────────────────────────
# SIMPLE TEST IMAGE
# ─────────────────────────────

def create_simple_image():
    img = np.ones((256, 256, 3), np.uint8) * 50
    cv2.circle(img, (80, 100), 50, (200, 200, 200), -1)
    cv2.rectangle(img, (140, 60), (220, 140), (100, 180, 220), -1)
    return img


# ─────────────────────────────
# RUN PROGRAM
# ─────────────────────────────

if __name__ == "__main__":
    import sys
    run_demo(sys.argv[1] if len(sys.argv) > 1 else None)