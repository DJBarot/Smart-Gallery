
import subprocess
import sys
from pathlib import Path

from src.database import get_connection
from src.similarity import find_similar_images, display_results
from src.text_search import search_by_text, display_text_results


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_MANAGER = PROJECT_ROOT / "src" / "image_manager.py"


def view_all_images():
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT image_id, filename, file_path
            FROM images
            ORDER BY image_id
        """)

        images = cursor.fetchall()

        print("\n===================================")
        print("           IMAGE GALLERY")
        print("===================================")

        if not images:
            print("No images found.")
            return

        for image_id, filename, file_path in images:
            print(f"\nID       : {image_id}")
            print(f"Filename : {filename}")
            print(f"Path     : {file_path}")

        print(f"\nTotal images: {len(images)}")

    except Exception as error:
        print(f"Could not retrieve images: {error}")

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def index_images():
    print("\n===================================")
    print("           INDEX IMAGES")
    print("===================================")

    if not IMAGE_MANAGER.is_file():
        print(f"Image manager not found: {IMAGE_MANAGER}")
        return

    try:
        result = subprocess.run(
            [sys.executable, str(IMAGE_MANAGER)],
            cwd=str(PROJECT_ROOT),
            check=False
        )

        if result.returncode == 0:
            print("\nImage indexing process finished.")
        else:
            print(f"Indexing failed with exit code {result.returncode}.")

    except Exception as error:
        print(f"Could not run image indexing: {error}")


def search_by_image():
    image_path = input(
        "Enter the query image path: "
    ).strip().strip('"')

    if not image_path:
        print("Please enter an image path.")
        return

    if not Path(image_path).is_file():
        print("Error: Image file not found.")
        return

    top_n_input = input(
        "Number of similar images (default 5): "
    ).strip()

    try:
        top_n = int(top_n_input) if top_n_input else 5

        if top_n < 1:
            print("Enter a number greater than zero.")
            return

        results = find_similar_images(image_path, top_n=top_n)

        if results:
            display_results(results)
        else:
            print("No matching images found. Try indexing images first.")

    except Exception as error:
        print(f"Image search failed: {error}")


def search_by_text_menu():
    query = input(
        "Describe the image you want to find: "
    ).strip()

    if not query:
        print("Please enter a search description.")
        return

    top_n_input = input(
        "Number of results (default 5): "
    ).strip()

    try:
        top_n = int(top_n_input) if top_n_input else 5

        if top_n < 1:
            print("Enter a number greater than zero.")
            return

        results = search_by_text(query, top_n=top_n)
        display_text_results(results)

    except Exception as error:
        print(f"Text search failed: {error}")


def main():
    while True:
        print("\n===================================")
        print("       SMART IMAGE GALLERY")
        print("===================================")
        print("1. View all images")
        print("2. Index images")
        print("3. Search using an image")
        print("4. Search using text")
        print("5. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            view_all_images()

        elif choice == "2":
            index_images()

        elif choice == "3":
            search_by_image()

        elif choice == "4":
            search_by_text_menu()

        elif choice == "5":
            print("Thank you for using Smart Image Gallery!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
