
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_DIR = os.path.join(BASE_DIR, "features", "clip")

GALLERY_DIR = r"D:\SmartGallery_Selected"
EVALUATION_DIR = r"D:\SmartGallery_Evaluation"

gallery_embeddings = np.load(
    os.path.join(FEATURE_DIR, "gallery_embeddings.npy")
)

gallery_colors = np.load(
    os.path.join(FEATURE_DIR, "gallery_color_histograms.npy")
)

gallery_metadata = pd.read_csv(
    os.path.join(FEATURE_DIR, "gallery_metadata.csv")
)

evaluation_embeddings = np.load(
    os.path.join(FEATURE_DIR, "evaluation_embeddings.npy")
)

evaluation_metadata = pd.read_csv(
    os.path.join(FEATURE_DIR, "evaluation_metadata.csv")
)

# Initial experimental weights
CLIP_WEIGHT = 0.7
COLOR_WEIGHT = 0.3

TOP_K = 10


# --------------------------------------------------
# COLOR SIMILARITY
# --------------------------------------------------

def calculate_color_similarity(query_histogram):

    # Histogram intersection: values between 0 and 1
    similarity = np.minimum(
        gallery_colors,
        query_histogram
    ).sum(axis=1)

    return similarity


# --------------------------------------------------
# SEARCH
# --------------------------------------------------

def search(query_index):

    query_embedding = evaluation_embeddings[query_index]

    query_row = evaluation_metadata.iloc[query_index]

    query_category = query_row["category"]
    query_filename = query_row["filename"]

    # Find the evaluation query image
    query_path = os.path.join(
        EVALUATION_DIR,
        query_category,
        query_filename
    )

    query_image = Image.open(query_path).convert("RGB")

    # Extract the query image color histogram
    from generate_color_features import extract_color_histogram

    query_histogram = extract_color_histogram(query_path)

    # CLIP similarity
    clip_scores = gallery_embeddings @ query_embedding

    # Color similarity
    color_scores = calculate_color_similarity(
        query_histogram
    )

    # Hybrid similarity
    hybrid_scores = (
        CLIP_WEIGHT * clip_scores +
        COLOR_WEIGHT * color_scores
    )

    # Rankings
    rankings = {
        "CLIP only": np.argsort(clip_scores)[::-1][:TOP_K],
        "Color only": np.argsort(color_scores)[::-1][:TOP_K],
        "Hybrid": np.argsort(hybrid_scores)[::-1][:TOP_K]
    }

    print("\nQuery:", query_filename)
    print("Expected category:", query_category)

    results = {}

    for method, indices in rankings.items():

        correct = np.sum(
            gallery_metadata.iloc[indices]["category"].values
            == query_category
        )

        precision = correct / TOP_K

        results[method] = precision

        print(
            f"{method:12} Precision@10: "
            f"{correct}/{TOP_K} = {precision * 100:.2f}%"
        )

    # Display results
    fig, axes = plt.subplots(
        3, 10,
        figsize=(20, 8)
    )

    fig.suptitle(
        f"Hybrid Search Comparison - {query_filename}",
        fontsize=15
    )

    for row_index, (method, indices) in enumerate(rankings.items()):

        for col_index, idx in enumerate(indices):

            row = gallery_metadata.iloc[idx]

            image_path = os.path.join(
                GALLERY_DIR,
                row["category"],
                row["filename"]
            )

            image = Image.open(image_path).convert("RGB")

            axes[row_index, col_index].imshow(image)

            axes[row_index, col_index].set_title(
                row["category"],
                fontsize=8
            )

            axes[row_index, col_index].axis("off")

        axes[row_index, 0].set_ylabel(
            method,
            fontsize=10
        )

    plt.tight_layout()
    plt.show()

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    # Start with one query image
    query_index = 0

    search(query_index)
