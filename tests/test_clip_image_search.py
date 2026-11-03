
from pathlib import Path

from src.text_search import search_by_image_clip


def main():
    image_folder = Path("images")

    # Use an existing gallery image as the query.
    query_image = image_folder / "glacier_0586.jpg"

    if not query_image.is_file():
        raise FileNotFoundError(query_image)

    results = search_by_image_clip(
        str(query_image),
        top_n=10
    )

    print("\n========== CLIP IMAGE SEARCH ==========")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['filename']} | "
            f"Similarity: {result['similarity_score']:.4f}"
        )

    print("=======================================")


if __name__ == "__main__":
    main()
