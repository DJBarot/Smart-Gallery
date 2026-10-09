import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from feature_extraction import (
    preprocess_image,
    extract_hsv_histogram,
    extract_dominant_colors
)


# ============================================
# IMAGE PATH
# ============================================

project_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

image_path = os.path.join(
    project_folder,
    "images",
    "straw-hat-luffy-one-3840x2160-22500.png"
)


# ============================================
# LOAD IMAGE
# ============================================

original_image = cv2.imread(image_path)

if original_image is None:
    raise ValueError(
        "Unable to load image: " + image_path
    )


# ============================================
# IMAGE PROCESSING
# ============================================

filtered_image, hsv_image = preprocess_image(
    image_path
)

histogram = extract_hsv_histogram(
    hsv_image
)

centers, percentages = extract_dominant_colors(
    hsv_image,
    k=3
)


# ============================================
# CONVERT IMAGES FOR DISPLAY
# ============================================

original_rgb = cv2.cvtColor(
    original_image,
    cv2.COLOR_BGR2RGB
)

filtered_rgb = cv2.cvtColor(
    filtered_image,
    cv2.COLOR_BGR2RGB
)


# ============================================
# CREATE FIGURE
# ============================================

plt.figure(figsize=(14, 9))


# ============================================
# ORIGINAL IMAGE
# ============================================

plt.subplot(2, 2, 1)

plt.imshow(original_rgb)

plt.title("Original Image")

plt.axis("off")


# ============================================
# PROCESSED IMAGE
# ============================================

plt.subplot(2, 2, 2)

plt.imshow(filtered_rgb)

plt.title("After Gaussian Filtering")

plt.axis("off")


# ============================================
# HSV HISTOGRAM
# ============================================

plt.subplot(2, 2, 3)

plt.bar(
    range(len(histogram)),
    histogram
)

plt.title("Normalized HSV Histogram")

plt.xlabel("Histogram Bin")

plt.ylabel("Normalized Frequency")


# ============================================
# DOMINANT COLORS
# ============================================

plt.subplot(2, 2, 4)

for i in range(3):

    hsv_color = np.uint8(
        [[centers[i]]]
    )

    rgb_color = cv2.cvtColor(
        hsv_color,
        cv2.COLOR_HSV2RGB
    )[0][0] / 255.0

    plt.bar(
        i,
        percentages[i],
        color=rgb_color
    )

    plt.text(
        i,
        percentages[i] + 0.02,
        str(round(percentages[i] * 100, 2)) + "%",
        ha="center"
    )


plt.title("Dominant Colors")

plt.xlabel("Color Cluster")

plt.ylabel("Percentage")

plt.xticks(
    [0, 1, 2],
    ["Color 1", "Color 2", "Color 3"]
)

plt.ylim(0, max(percentages) + 0.15)


# ============================================
# FINAL DISPLAY
# ============================================

plt.tight_layout()

plt.show()