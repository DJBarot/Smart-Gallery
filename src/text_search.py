
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.database import get_connection
from src.embedding_manager import load_embedding_store


# ============================================
# CLIP CONFIGURATION
# ============================================

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_TEXT_SIMILARITY = 0.23

_model = None
_processor = None

_gallery_embeddings = None
_gallery_metadata = None


# ============================================
# LOAD CLIP MODEL
# ============================================

def load_model():
    """Load CLIP once and reuse it."""

    global _model, _processor

    if _model is None or _processor is None:
        print(f"Loading CLIP model on {DEVICE}...")

        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model = CLIPModel.from_pretrained(MODEL_NAME)

        _model.to(DEVICE)
        _model.eval()

        print("CLIP model loaded successfully.")

    return _model, _processor


# ============================================
# LOAD PERSISTENT GALLERY EMBEDDINGS
# ============================================

def load_gallery_embeddings():
    """
    Load the persistent gallery embedding matrix and metadata.

    Gallery images are not encoded again during searches.
    """

    global _gallery_embeddings, _gallery_metadata

    if _gallery_embeddings is None or _gallery_metadata is None:

        print("\nLoading persistent gallery embeddings...")

        embeddings, metadata = load_embedding_store()

        required_fields = {
            "image_id",
            "filename",
            "image_path"
        }

        for item in metadata:
            if not required_fields.issubset(item.keys()):
                raise ValueError(
                    "Embedding metadata is missing image_id, "
                    "filename or image_path. Run the embedding "
                    "synchronization script again."
                )

        image_ids = [int(item["image_id"]) for item in metadata]

        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Duplicate IMAGE_ID values in embedding metadata.")

        _gallery_embeddings = embeddings.astype(
            np.float32,
            copy=False
        )

        _gallery_metadata = metadata

        print(
            "Persistent gallery loaded:",
            len(_gallery_metadata),
            "images"
        )

        print(
            "Embedding matrix:",
            _gallery_embeddings.shape
        )

    return _gallery_embeddings, _gallery_metadata

def refresh_embedding_cache():
    """Clear cached gallery embeddings so the next search reloads them."""
    global _gallery_embeddings, _gallery_metadata

    _gallery_embeddings = None
    _gallery_metadata = None

    print("CLIP embedding cache cleared.")

# ============================================
# ORACLE IMAGE RETRIEVAL
# ============================================

def get_gallery_images():
    """Retrieve image metadata from Oracle."""

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            SELECT image_id, filename, file_path
            FROM images
            ORDER BY image_id
        """)

        return cursor.fetchall()

    finally:
        cursor.close()
        connection.close()


# ============================================
# QUERY IMAGE EMBEDDING
# ============================================

def get_image_embedding(file_path, model, processor):
    """
    Generate an embedding for a query image.

    This function is used for the uploaded/query image only.
    Gallery embeddings come from the persistent matrix.
    """

    absolute_path = os.path.abspath(file_path)

    if not os.path.isfile(absolute_path):
        raise FileNotFoundError(
            f"Image not found: {absolute_path}"
        )

    with Image.open(absolute_path) as source:
        image = source.convert("RGB")

    try:
        image_inputs = processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = image_inputs["pixel_values"].to(DEVICE)

        with torch.inference_mode():
            image_output = model.vision_model(
                pixel_values=pixel_values
            )

            image_features = model.visual_projection(
                image_output.pooler_output
            )

            image_features = torch.nn.functional.normalize(
                image_features.float(),
                p=2,
                dim=-1
            )

        return image_features[0].cpu().numpy()

    finally:
        image.close()


# ============================================
# RESULT GENERATION
# ============================================

def build_search_results(scores, metadata, top_n):
    """Convert similarity scores into the GUI-compatible result format."""

    top_n = min(top_n, len(scores))

    # Get indices in descending similarity order.
    ranked_indices = np.argsort(-scores)[:top_n]

    results = []

    for index in ranked_indices:
        item = metadata[int(index)]

        results.append({
            "image_id": int(item["image_id"]),
            "filename": item["filename"],
            "file_path": item["image_path"],
            "similarity_score": float(scores[index])
        })

    return results


# ============================================
# TEXT-BASED IMAGE SEARCH
# ============================================

def search_by_text(query, top_n=5):
    """
    Search images using a natural-language description.

    Only the text query is encoded.
    Gallery embeddings are loaded from the persistent store.
    """

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if top_n < 1:
        raise ValueError("top_n must be greater than zero.")

    embeddings, metadata = load_gallery_embeddings()
    model, processor = load_model()

    text_inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    text_inputs = {
        key: value.to(DEVICE)
        for key, value in text_inputs.items()
    }

    with torch.inference_mode():
        text_output = model.text_model(**text_inputs)

        text_features = model.text_projection(
            text_output.pooler_output
        )

        text_features = torch.nn.functional.normalize(
            text_features.float(),
            p=2,
            dim=-1
        )

    query_vector = text_features[0].cpu().numpy()

    # One matrix multiplication compares against the full gallery.
    scores = embeddings @ query_vector

    # Apply the existing relevance threshold.
    relevant_indices = np.where(
        scores >= MIN_TEXT_SIMILARITY
    )[0]

    if len(relevant_indices) == 0:
        print("No sufficiently relevant images found.")
        return []

    relevant_scores = scores[relevant_indices]

    results = build_search_results(
        relevant_scores,
        [metadata[i] for i in relevant_indices],
        top_n
    )

    print(f"\nText search: {query}")
    print(f"Images searched: {len(embeddings)}")
    print(f"Relevant images: {len(relevant_indices)}")

    return results


# ============================================
# DISPLAY TEXT SEARCH RESULTS
# ============================================

def display_text_results(results):
    print("\n===================================")
    print("       TEXT SEARCH RESULTS")
    print("===================================")

    if not results:
        print("No searchable images found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Image ID: {result['image_id']}")
        print(f"Filename: {result['filename']}")

        print(
            "Cosine similarity:",
            round(result["similarity_score"], 4)
        )

        print(f"Path: {result['file_path']}")


# ============================================
# IMAGE-BASED SEMANTIC SEARCH
# ============================================

def search_by_image_clip(image_path, top_n=5):
    """
    Search the gallery using CLIP semantic image similarity.

    Only the query image is encoded.
    Gallery embeddings are loaded from the persistent store.
    """

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Query image not found: {image_path}"
        )

    if top_n < 1:
        raise ValueError("top_n must be greater than zero.")

    embeddings, metadata = load_gallery_embeddings()
    model, processor = load_model()

    query_embedding = get_image_embedding(
        image_path,
        model,
        processor
    )

    # Compare the query vector with every stored gallery vector.
    scores = embeddings @ query_embedding

    results = build_search_results(
        scores,
        metadata,
        top_n
    )

    print("\nCLIP semantic image search completed.")
    print("Images searched:", len(embeddings))

    return results


# ============================================
# STANDALONE TEST
# ============================================

if __name__ == "__main__":

    query = input(
        "Enter a description of the image you want to find: "
    ).strip()

    top_n_input = input(
        "Number of results (default 5): "
    ).strip()

    try:
        top_n = int(top_n_input) if top_n_input else 5

        results = search_by_text(
            query,
            top_n=top_n
        )

        display_text_results(results)

    except (ValueError, RuntimeError, OSError) as error:
        print(f"Text search failed: {error}")
