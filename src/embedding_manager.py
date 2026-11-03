
import csv
from pathlib import Path

import numpy as np


# ============================================
# CONFIGURATION
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLIP_FOLDER = PROJECT_ROOT / "features" / "clip"

EMBEDDINGS_FILE = CLIP_FOLDER / "gallery_embeddings.npy"
METADATA_FILE = CLIP_FOLDER / "gallery_metadata.csv"

EXPECTED_DIMENSION = 512
MODEL_NAME = "openai/clip-vit-base-patch32"


# ============================================
# LOAD EXISTING EMBEDDING STORE
# ============================================

def load_embedding_store():
    """
    Load existing CLIP embeddings and their metadata.

    This function does not generate embeddings or load the CLIP model.
    """

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    print("\nLoading existing CLIP embedding store...")

    embeddings = np.load(
        EMBEDDINGS_FILE,
        allow_pickle=False
    )

    metadata = []

    with open(
        METADATA_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "filename",
            "category",
            "image_path"
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "Metadata CSV is missing required columns."
            )

        metadata = list(reader)

    validate_embedding_store(embeddings, metadata)

    print("\nExisting embedding store loaded successfully.")
    print("Total embeddings:", len(embeddings))
    print("Embedding dimension:", embeddings.shape[1])

    return embeddings, metadata


# ============================================
# VALIDATE EMBEDDING STORE
# ============================================

def validate_embedding_store(embeddings, metadata):
    """
    Validate embedding dimensions, metadata count,
    numerical values and vector normalization.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got {embeddings.shape}"
        )

    if embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION}-dimensional embeddings, "
            f"got {embeddings.shape[1]}"
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding count and metadata count do not match: "
            f"{len(embeddings)} vs {len(metadata)}"
        )

    if len(embeddings) == 0:
        raise ValueError("Embedding store is empty.")

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embedding matrix contains NaN or infinite values."
        )

    # CLIP embeddings were L2-normalized during generation.
    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "Some embeddings are not L2-normalized."
        )

    filenames = [
        item["filename"]
        for item in metadata
    ]

    if len(filenames) != len(set(filenames)):
        raise ValueError(
            "Duplicate filenames found in embedding metadata."
        )


# ============================================
# DISPLAY EMBEDDING STORE INFORMATION
# ============================================

def show_embedding_store_info(embeddings, metadata):

    print("\n========================================")
    print("       CLIP EMBEDDING STORE")
    print("========================================")

    print("Model:", MODEL_NAME)
    print("Embedding file:", EMBEDDINGS_FILE)
    print("Metadata file:", METADATA_FILE)

    print("Total images:", len(metadata))
    print("Embedding shape:", embeddings.shape)

    categories = {}

    for item in metadata:
        category = item["category"]

        categories[category] = (
            categories.get(category, 0) + 1
        )

    print("\nCategory distribution:")

    for category, count in sorted(categories.items()):
        print(f"{category}: {count}")

    missing_paths = sum(
        1
        for item in metadata
        if not Path(item["image_path"]).is_file()
    )

    print("\nMetadata paths not currently accessible:", missing_paths)

    print("\n========================================")


# ============================================
# MAIN TEST
# ============================================

if __name__ == "__main__":

    embeddings, metadata = load_embedding_store()

    show_embedding_store_info(
        embeddings,
        metadata
    )

    print("\nNo new embeddings generated.")
    print("Existing embeddings were only loaded and validated.")
