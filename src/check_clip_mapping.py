
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from src.database import get_connection
from src.embedding_manager import METADATA_FILE


def sha256_file(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    # Load existing CLIP metadata.
    with open(METADATA_FILE, "r", newline="", encoding="utf-8") as file:
        metadata = list(csv.DictReader(file))

    # Retrieve all Oracle image records.
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            SELECT image_id, filename, file_path
            FROM images
            ORDER BY image_id
        """)
        database_images = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    # Index Oracle records by filename.
    database_by_filename = defaultdict(list)

    for image_id, filename, file_path in database_images:
        database_by_filename[filename].append(
            (image_id, file_path)
        )

    matched = 0
    missing_database_record = 0
    missing_source_file = 0
    missing_database_file = 0
    ambiguous = 0
    content_mismatch = 0

    print("\nChecking CLIP metadata against Oracle...\n")

    for item in metadata:
        filename = item["filename"]
        source_path = Path(item["image_path"])

        if not source_path.is_file():
            missing_source_file += 1
            continue

        records = database_by_filename.get(filename, [])

        if not records:
            missing_database_record += 1
            continue

        if len(records) > 1:
            ambiguous += 1
            continue

        image_id, database_path = records[0]
        database_path = Path(database_path)

        if not database_path.is_file():
            missing_database_file += 1
            continue

        source_hash = sha256_file(source_path)
        database_hash = sha256_file(database_path)

        if source_hash == database_hash:
            matched += 1
        else:
            content_mismatch += 1
            print(f"CONTENT MISMATCH: {filename} (ID: {image_id})")

    print("\n========== MAPPING REPORT ==========")
    print("CLIP metadata records:", len(metadata))
    print("Oracle image records:", len(database_images))
    print("Verified hash matches:", matched)
    print("Missing Oracle records:", missing_database_record)
    print("Missing source files:", missing_source_file)
    print("Missing Oracle files:", missing_database_file)
    print("Ambiguous filenames:", ambiguous)
    print("Content mismatches:", content_mismatch)
    print("====================================")


if __name__ == "__main__":
    main()
