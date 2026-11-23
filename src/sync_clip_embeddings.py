import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.database import get_connection
from src.embedding_manager import (
    EMBEDDINGS_FILE,
    METADATA_FILE,
    MODEL_NAME,
    EXPECTED_DIMENSION,
    load_embedding_store,
)

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_database_images():
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


def encode_images(paths, model, processor):
    """Generate normalized CLIP image embeddings in batches."""
    all_embeddings = []

    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        images = []
        try:
            for path in batch_paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))

            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(DEVICE)

            with torch.inference_mode():
                output = model.vision_model(pixel_values=pixel_values)
                features = model.visual_projection(output.pooler_output)
                features = torch.nn.functional.normalize(
                    features.float(), p=2, dim=-1
                )

            all_embeddings.append(features.cpu().numpy())
        finally:
            for image in images:
                image.close()

        print(
            f"Encoded {min(start + BATCH_SIZE, len(paths))}"
            f"/{len(paths)} images needing embeddings"
        )

    if not all_embeddings:
        return np.empty((0, EXPECTED_DIMENSION), dtype=np.float32)
    return np.concatenate(all_embeddings, axis=0)


def atomic_save(embeddings, metadata):
    """Write both persistent files safely using temporary files."""
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    embedding_temp = metadata_temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=EMBEDDINGS_FILE.parent,
            suffix=".tmp", delete=False
        ) as file:
            embedding_temp = Path(file.name)
            np.save(file, embeddings, allow_pickle=False)

        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8",
            dir=METADATA_FILE.parent, suffix=".tmp", delete=False
        ) as file:
            metadata_temp = Path(file.name)
            fields = ["image_id", "filename", "category", "image_path"]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metadata)

        os.replace(embedding_temp, EMBEDDINGS_FILE)
        embedding_temp = None
        os.replace(metadata_temp, METADATA_FILE)
        metadata_temp = None
    finally:
        for temp in (embedding_temp, metadata_temp):
            if temp and temp.exists():
                temp.unlink()


def main():
    print("\nLoading existing CLIP embedding store...")
    embeddings, old_metadata = load_embedding_store()

    print("\nReading Oracle image records...")
    database_images = get_database_images()
    print("Oracle records:", len(database_images))
    print("Existing CLIP embeddings:", len(embeddings))

    # Key current Oracle records by IMAGE_ID. Filenames are not reliable
    # identifiers because files can be renamed or replaced.
    database_by_id = {}
    for image_id, filename, file_path in database_images:
        image_id = int(image_id)
        if image_id in database_by_id:
            raise RuntimeError(f"Duplicate Oracle IMAGE_ID: {image_id}")
        if not Path(file_path).is_file():
            raise FileNotFoundError(
                f"Oracle image file is missing for {filename}: {file_path}"
            )
        database_by_id[image_id] = (filename, str(Path(file_path).resolve()))

    # Reuse an old vector only if its IMAGE_ID, filename and path still
    # identify the same gallery item. Deleted/stale records are discarded.
    vectors_by_id = {}
    metadata_by_id = {}
    reused_old_indices = set()
    legacy_by_filename = {
        item["filename"]: index
        for index, item in enumerate(old_metadata)
        if not item.get("image_id")
    }

    for index, item in enumerate(old_metadata):
        raw_id = item.get("image_id")
        if not raw_id:
            continue
        image_id = int(raw_id)
        current = database_by_id.get(image_id)
        if current is None:
            print(f"Discarding deleted CLIP entry: {item['filename']}")
            continue

        current_filename, current_path = current
        old_path = str(Path(item.get("image_path", "")).resolve())
        if item["filename"] == current_filename and old_path == current_path:
            vectors_by_id[image_id] = embeddings[index]
            reused_old_indices.add(index)
            metadata_by_id[image_id] = {
                "image_id": image_id,
                "filename": current_filename,
                "category": item.get("category", ""),
                "image_path": current_path,
            }

    # Backward compatibility for any older metadata without IMAGE_ID.
    for image_id, (filename, file_path) in database_by_id.items():
        if image_id in vectors_by_id:
            continue
        old_index = legacy_by_filename.get(filename)
        if old_index is not None:
            old_path = str(Path(old_metadata[old_index]["image_path"]).resolve())
            if old_path == file_path:
                vectors_by_id[image_id] = embeddings[old_index]
                reused_old_indices.add(old_index)
                metadata_by_id[image_id] = {
                    "image_id": image_id,
                    "filename": filename,
                    "category": old_metadata[old_index].get("category", ""),
                    "image_path": file_path,
                }

    missing = [
        (image_id, filename, file_path)
        for image_id, (filename, file_path) in database_by_id.items()
        if image_id not in vectors_by_id
    ]

    print("Existing embeddings reused:", len(vectors_by_id))
    print("New/replacement embeddings required:", len(missing))
    print("Stale/deleted embeddings removed:", len(old_metadata) - len(reused_old_indices))

    if missing:
        print(f"\nLoading CLIP model on {DEVICE}...")
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()
        try:
            new_vectors = encode_images(
                [record[2] for record in missing], model, processor
            )
        finally:
            del model
            del processor

        if new_vectors.shape != (len(missing), EXPECTED_DIMENSION):
            raise RuntimeError(f"Unexpected embedding shape: {new_vectors.shape}")

        for index, (image_id, filename, file_path) in enumerate(missing):
            vectors_by_id[image_id] = new_vectors[index]
            metadata_by_id[image_id] = {
                "image_id": image_id,
                "filename": filename,
                "category": "",
                "image_path": file_path,
            }

    ordered_ids = sorted(database_by_id)
    final_metadata = [metadata_by_id[i] for i in ordered_ids]
    if ordered_ids:
        final_embeddings = np.stack(
            [vectors_by_id[i] for i in ordered_ids]
        ).astype(np.float32)
    else:
        final_embeddings = np.empty((0, EXPECTED_DIMENSION), dtype=np.float32)

    if final_embeddings.shape != (len(database_images), EXPECTED_DIMENSION):
        raise RuntimeError(f"Final embedding shape mismatch: {final_embeddings.shape}")
    if not np.isfinite(final_embeddings).all():
        raise RuntimeError("Non-finite values found in embeddings.")
    if len(final_embeddings):
        norms = np.linalg.norm(final_embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise RuntimeError("Embedding normalization validation failed.")

    atomic_save(final_embeddings, final_metadata)
    print("\nCLIP synchronization completed successfully.")
    print("Oracle images:", len(database_images))
    print("Final CLIP embeddings:", len(final_metadata))
    print("Newly generated embeddings:", len(missing))


if __name__ == "__main__":
    main()
