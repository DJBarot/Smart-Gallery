import os
import json
import cv2

from feature_extraction import (
    preprocess_image,
    extract_hsv_histogram,
    extract_dominant_colors
)

from database import get_connection


def insert_image(image_path):
    """
    Process one image and store/update its metadata
    and extracted features in Oracle.
    """

    # -----------------------------------------
    # 1. Check image exists
    # -----------------------------------------
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    filename = os.path.basename(image_path)

    # -----------------------------------------
    # 2. Read ORIGINAL image
    # -----------------------------------------
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Unable to load image:", image_path)
        return

    original_height, original_width = original_image.shape[:2]

    print("Original image size:",
          original_width, "x", original_height)

    # -----------------------------------------
    # 3. Process image for feature extraction
    # -----------------------------------------
    filtered_image, hsv_image = preprocess_image(image_path)

    # -----------------------------------------
    # 4. Extract features
    # -----------------------------------------
    histogram = extract_hsv_histogram(hsv_image)

    dominant_colors, percentages = extract_dominant_colors(
        hsv_image,
        k=3
    )

    # -----------------------------------------
    # 5. Get image format
    # -----------------------------------------
    image_format = os.path.splitext(filename)[1].replace(".", "").upper()

    # -----------------------------------------
    # 6. Connect to Oracle
    # -----------------------------------------
    connection = get_connection()
    cursor = connection.cursor()

    try:

        # -----------------------------------------
        # 7. Check if image already exists
        # -----------------------------------------
        cursor.execute(
            """
            SELECT image_id
            FROM images
            WHERE filename = :filename
            """,
            {
                "filename": filename
            }
        )

        existing_image = cursor.fetchone()

        # -----------------------------------------
        # 8. Insert or update IMAGES
        # -----------------------------------------
        if existing_image is None:

            image_id_var = cursor.var(int)

            cursor.execute(
                """
                INSERT INTO images
                (
                    filename,
                    file_path,
                    width,
                    height,
                    image_format
                )
                VALUES
                (
                    :filename,
                    :file_path,
                    :width,
                    :height,
                    :image_format
                )
                RETURNING image_id INTO :image_id
                """,
                {
                    "filename": filename,
                    "file_path": os.path.abspath(image_path),
                    "width": original_width,
                    "height": original_height,
                    "image_format": image_format,
                    "image_id": image_id_var
                }
            )

            image_id = image_id_var.getvalue()[0]

            print("New image inserted.")
            print("Image ID:", image_id)

        else:

            image_id = existing_image[0]

            cursor.execute(
                """
                UPDATE images
                SET
                    file_path = :file_path,
                    width = :width,
                    height = :height,
                    image_format = :image_format
                WHERE image_id = :image_id
                """,
                {
                    "file_path": os.path.abspath(image_path),
                    "width": original_width,
                    "height": original_height,
                    "image_format": image_format,
                    "image_id": image_id
                }
            )

            print("Existing image updated.")
            print("Image ID:", image_id)

        # -----------------------------------------
        # 9. Prepare dominant colors
        # -----------------------------------------
        color1_h = float(dominant_colors[0][0]) / 180.0
        color1_s = float(dominant_colors[0][1]) / 255.0
        color1_v = float(dominant_colors[0][2]) / 255.0
        color1_percentage = float(percentages[0])

        color2_h = float(dominant_colors[1][0]) / 180.0
        color2_s = float(dominant_colors[1][1]) / 255.0
        color2_v = float(dominant_colors[1][2]) / 255.0
        color2_percentage = float(percentages[1])

        color3_h = float(dominant_colors[2][0]) / 180.0
        color3_s = float(dominant_colors[2][1]) / 255.0
        color3_v = float(dominant_colors[2][2]) / 255.0
        color3_percentage = float(percentages[2])

        # -----------------------------------------
        # 10. Convert histogram to JSON
        # -----------------------------------------
        histogram_json = json.dumps(histogram.tolist())

        # -----------------------------------------
        # 11. Check IMAGE_FEATURES
        # -----------------------------------------
        cursor.execute(
            """
            SELECT feature_id
            FROM image_features
            WHERE image_id = :image_id
            """,
            {
                "image_id": image_id
            }
        )

        existing_features = cursor.fetchone()

        # -----------------------------------------
        # 12. Insert or update IMAGE_FEATURES
        # -----------------------------------------
        if existing_features is None:

            cursor.execute(
                """
                INSERT INTO image_features
                (
                    image_id,

                    color1_h,
                    color1_s,
                    color1_v,
                    color1_percentage,

                    color2_h,
                    color2_s,
                    color2_v,
                    color2_percentage,

                    color3_h,
                    color3_s,
                    color3_v,
                    color3_percentage,

                    histogram_data
                )
                VALUES
                (
                    :image_id,

                    :color1_h,
                    :color1_s,
                    :color1_v,
                    :color1_percentage,

                    :color2_h,
                    :color2_s,
                    :color2_v,
                    :color2_percentage,

                    :color3_h,
                    :color3_s,
                    :color3_v,
                    :color3_percentage,

                    :histogram_data
                )
                """,
                {
                    "image_id": image_id,

                    "color1_h": color1_h,
                    "color1_s": color1_s,
                    "color1_v": color1_v,
                    "color1_percentage": color1_percentage,

                    "color2_h": color2_h,
                    "color2_s": color2_s,
                    "color2_v": color2_v,
                    "color2_percentage": color2_percentage,

                    "color3_h": color3_h,
                    "color3_s": color3_s,
                    "color3_v": color3_v,
                    "color3_percentage": color3_percentage,

                    "histogram_data": histogram_json
                }
            )

            print("Image features inserted.")

        else:

            cursor.execute(
                """
                UPDATE image_features
                SET
                    color1_h = :color1_h,
                    color1_s = :color1_s,
                    color1_v = :color1_v,
                    color1_percentage = :color1_percentage,

                    color2_h = :color2_h,
                    color2_s = :color2_s,
                    color2_v = :color2_v,
                    color2_percentage = :color2_percentage,

                    color3_h = :color3_h,
                    color3_s = :color3_s,
                    color3_v = :color3_v,
                    color3_percentage = :color3_percentage,

                    histogram_data = :histogram_data
                WHERE image_id = :image_id
                """,
                {
                    "image_id": image_id,

                    "color1_h": color1_h,
                    "color1_s": color1_s,
                    "color1_v": color1_v,
                    "color1_percentage": color1_percentage,

                    "color2_h": color2_h,
                    "color2_s": color2_s,
                    "color2_v": color2_v,
                    "color2_percentage": color2_percentage,

                    "color3_h": color3_h,
                    "color3_s": color3_s,
                    "color3_v": color3_v,
                    "color3_percentage": color3_percentage,

                    "histogram_data": histogram_json
                }
            )

            print("Image features updated.")

        # -----------------------------------------
        # 13. Commit
        # -----------------------------------------
        connection.commit()

        print("===================================")
        print("IMAGE PROCESSING COMPLETE")
        print("===================================")
        print("Filename:", filename)
        print("Original size:",
              original_width, "x", original_height)
        print("Histogram features:", len(histogram))
        print("Dominant colors: 3")
        print("Total feature vector: 268")

    except Exception as error:

        connection.rollback()

        print("Database operation failed.")
        print(error)

    finally:

        cursor.close()
        connection.close()

        print("Database connection closed.")


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    project_folder = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    image_folder = os.path.join(
        project_folder,
        "images"
    )

    image_files = [
        "djb passport.jpg",
        "Screenshot 2026-06-21 144345.jpg",
        "straw-hat-luffy-one-3840x2160-22500.png"
    ]

    for filename in image_files:

        print()
        print("===================================")
        print("PROCESSING:", filename)
        print("===================================")

        image_path = os.path.join(
            image_folder,
            filename
        )

        insert_image(image_path)