import cv2
import numpy as np
from sklearn.cluster import KMeans


# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(image_path):
    """
    Load and preprocess an image.

    Steps:
    1. Read image
    2. Resize to 256 x 256
    3. Apply Gaussian filter
    4. Convert BGR to HSV
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to load image: " + image_path)

    # Resize image
    image = cv2.resize(image, (256, 256))

    # Gaussian filtering
    filtered_image = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # Convert BGR to HSV
    hsv_image = cv2.cvtColor(
        filtered_image,
        cv2.COLOR_BGR2HSV
    )

    return filtered_image, hsv_image


# ============================================
# HSV HISTOGRAM
# ============================================

def extract_hsv_histogram(hsv_image):
    """
    Extract a 16 x 4 x 4 HSV histogram.

    H = 16 bins
    S = 4 bins
    V = 4 bins

    Total = 256 features
    """

    histogram = cv2.calcHist(
        [hsv_image],
        [0, 1, 2],
        None,
        [16, 4, 4],
        [0, 180, 0, 256, 0, 256]
    )

    # Flatten 3D histogram into 1D vector
    histogram = histogram.flatten()

    # Normalize histogram
    histogram = histogram / histogram.sum()

    return histogram


# ============================================
# K-MEANS DOMINANT COLORS
# ============================================

def extract_dominant_colors(hsv_image, k=3):
    """
    Extract 3 dominant colors using K-Means.

    Returns:
        3 dominant HSV colors
        Percentage of pixels belonging to each color
    """

    # Convert image into list of HSV pixels
    pixels = hsv_image.reshape((-1, 3))

    # Convert to float
    pixels = np.float32(pixels)

    # K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(pixels)

    centers = kmeans.cluster_centers_

    # Count pixels in each cluster
    counts = np.bincount(labels)

    # Calculate percentages
    percentages = counts / len(labels)

    # Sort clusters from largest to smallest
    order = np.argsort(percentages)[::-1]

    centers = centers[order]
    percentages = percentages[order]

    return centers, percentages


# ============================================
# CREATE FINAL FEATURE VECTOR
# ============================================

def create_feature_vector(hsv_image):

    # 256 histogram features
    histogram = extract_hsv_histogram(hsv_image)

    # 3 dominant colors
    centers, percentages = extract_dominant_colors(
        hsv_image,
        k=3
    )

    dominant_features = []

    for i in range(3):

        h = centers[i][0] / 180.0
        s = centers[i][1] / 255.0
        v = centers[i][2] / 255.0
        percentage = percentages[i]

        dominant_features.extend([
            h,
            s,
            v,
            percentage
        ])

    # Combine histogram + dominant colors
    feature_vector = np.concatenate([
        histogram,
        np.array(dominant_features)
    ])

    return feature_vector, centers, percentages


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    import os

    project_folder = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    image_path = os.path.join(
        project_folder,
        "images",
        "djb passport.jpg"
    )

    filtered_image, hsv_image = preprocess_image(
        image_path
    )

    feature_vector, centers, percentages = create_feature_vector(
        hsv_image
    )

    print("\n===================================")
    print("SMART IMAGE GALLERY")
    print("FEATURE EXTRACTION TEST")
    print("===================================")

    print("\nImage:")
    print(image_path)

    print("\nImage Size:")
    print("256 x 256")

    print("\nHistogram Features:")
    print(len(feature_vector) - 12)

    print("\nDominant Colors:")

    for i in range(3):

        print(
            "\nColor", i + 1,
            "\nHSV:",
            centers[i],
            "\nPercentage:",
            round(percentages[i] * 100, 2),
            "%"
        )

    print("\nTotal Feature Vector Size:")
    print(len(feature_vector))

    print("\nFirst 10 Histogram Values:")
    print(feature_vector[:10])

    print("\n===================================")