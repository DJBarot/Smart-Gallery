
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

        for path in batch_paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.inference_mode():
            output = model.vision_model(
                pixel_values=pixel_values
            )

            features = model.visual_projection(
                output.pooler_output
            )

            features = torch.nn.functional.normalize(
                features.float(),
                p=2,
                dim=-1
            )

        all_embeddings.append(
            features.cpu().numpy()
        )

        for image in images:
            image.close()

        print(
            f"Encoded {min(start + BATCH_SIZE, len(paths))}"
            f"/{len(paths)} missing images"
        )

    if not all_embeddings:
        return np.empty((0, EXPECTED_DIMENSION), dtype=np.float32)

    return np.concatenate(all_embeddings, axis=0)


def atomic_save(embeddings, metadata):
    """Save both files using temporary files before replacing originals."""

    embedding_temp = None
    metadata_temp = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=EMBEDDINGS_FILE.parent,
            suffix=".tmp",
            delete=False
        ) as file:
            embedding_temp = Path(file.name)
            np.save(file, embeddings, allow_pickle=False)

        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=METADATA_FILE.parent,
            suffix=".tmp",
            delete=False
        ) as file:
            metadata_temp = Path(file.name)

            fieldnames = [
                "image_id",
                "filename",
                "category",
                "image_path"
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(metadata)

        # Replace only after both temporary files were written.
        os.replace(embedding_temp, EMBEDDINGS_FILE)
        embedding_temp = None

        os.replace(metadata_temp, METADATA_FILE)
        metadata_temp = None

    finally:
        if embedding_temp and embedding_temp.exists():
            embedding_temp.unlink()

        if metadata_temp and metadata_temp.exists():
            metadata_temp.unlink()


def main():
    print("\nLoading existing embedding store...")

    embeddings, old_metadata = load_embedding_store()

    print("\nReading Oracle image records...")
    database_images = get_database_images()

    print("Oracle records:", len(database_images))
    print("Existing embeddings:", len(embeddings))

    if len(database_images) < len(embeddings):
        raise RuntimeError(
            "Oracle has fewer images than the existing embedding store."
        )

    # Index existing metadata by filename.
    metadata_by_filename = {
        item["filename"]: item
        for item in old_metadata
    }

    database_by_filename = {}

    for image_id, filename, file_path in database_images:
        if filename in database_by_filename:
            raise RuntimeError(
                f"Duplicate Oracle filename detected: {filename}"
            )

        if not Path(file_path).is_file():
            raise FileNotFoundError(
                f"Oracle image file is missing: {file_path}"
            )

        database_by_filename[filename] = (
            image_id,
            file_path
        )

    # Match existing vectors to Oracle IDs.
    matched_embeddings = {}
    final_metadata = []
    missing_records = []

    for index, item in enumerate(old_metadata):
        filename = item["filename"]

        if filename not in database_by_filename:
            raise RuntimeError(
                f"No Oracle record found for: {filename}"
            )

        image_id, file_path = database_by_filename[filename]

        matched_embeddings[image_id] = embeddings[index]

        final_metadata.append({
            "image_id": image_id,
            "filename": filename,
            "category": item.get("category", ""),
            "image_path": file_path
        })

    # Find images that do not yet have persistent embeddings.
    for image_id, filename, file_path in database_images:
        if image_id not in matched_embeddings:
            missing_records.append(
                (image_id, filename, file_path)
            )

    print("\nExisting embeddings matched:", len(matched_embeddings))
    print("New embeddings required:", len(missing_records))

    if len(matched_embeddings) + len(missing_records) != len(database_images):
        raise RuntimeError("Image mapping count validation failed.")

    # Generate embeddings only for missing images.
    if missing_records:
        print(f"\nLoading CLIP model on {DEVICE}...")

        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPModel.from_pretrained(MODEL_NAME)

        model.to(DEVICE)
        model.eval()

        missing_paths = [
            record[2]
            for record in missing_records
        ]

        new_embeddings = encode_images(
            missing_paths,
            model,
            processor
        )

        if new_embeddings.shape != (
            len(missing_records),
            EXPECTED_DIMENSION
        ):
            raise RuntimeError(
                f"Unexpected new embedding shape: {new_embeddings.shape}"
            )

        for index, (image_id, filename, file_path) in enumerate(
            missing_records
        ):
            matched_embeddings[image_id] = new_embeddings[index]

            # Category is intentionally blank for these images.
            # We do not infer categories from filenames.
            final_metadata.append({
                "image_id": image_id,
                "filename": filename,
                "category": "",
                "image_path": file_path
            })

        del model
        del processor

    # Reorder vectors and metadata by Oracle IMAGE_ID.
    final_metadata.sort(key=lambda item: item["image_id"])

    final_embeddings = np.stack([
        matched_embeddings[item["image_id"]]
        for item in final_metadata
    ]).astype(np.float32)

    # Final validation.
    if final_embeddings.shape != (
        len(database_images),
        EXPECTED_DIMENSION
    ):
        raise RuntimeError(
            f"Unexpected final embedding shape: {final_embeddings.shape}"
        )

    if not np.isfinite(final_embeddings).all():
        raise RuntimeError("Non-finite values found in embeddings.")

    norms = np.linalg.norm(final_embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("Embedding normalization validation failed.")

    if len({item["image_id"] for item in final_metadata}) != len(database_images):
        raise RuntimeError("Duplicate image IDs in final metadata.")

    print("\nAll validations passed.")
    print("Final embedding shape:", final_embeddings.shape)

    atomic_save(final_embeddings, final_metadata)

    print("\nPersistent CLIP store updated successfully.")
    print("Total images:", len(final_metadata))
    print("Newly generated embeddings:", len(missing_records))
    print("Existing embeddings reused:", len(old_metadata))


if __name__ == "__main__":
    main()
