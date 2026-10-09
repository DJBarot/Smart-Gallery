import os
import json
import numpy as np

from feature_extraction import (
    preprocess_image,
    extract_hsv_histogram,
    extract_dominant_colors
)

from database import get_connection


# ==================================================
# FEATURE EXTRACTION FOR QUERY IMAGE
# ==================================================

def extract_query_features(image_path):
    """
    Extract the same 268-dimensional features
    used for images stored in Oracle.
    """

    filtered_image, hsv_image = preprocess_image(image_path)

    # 256 histogram features
    histogram = extract_hsv_histogram(hsv_image)

    # 3 dominant colors
    dominant_colors, percentages = extract_dominant_colors(
        hsv_image,
        k=3
    )

    # Normalize dominant colors
    dominant_features = []

    for i in range(3):

        h = float(dominant_colors[i][0]) / 180.0
        s = float(dominant_colors[i][1]) / 255.0
        v = float(dominant_colors[i][2]) / 255.0
        percentage = float(percentages[i])

        dominant_features.extend([
            h,
            s,
            v,
            percentage
        ])

    return histogram, np.array(dominant_features)


# ==================================================
# HISTOGRAM DISTANCE
# ==================================================

def calculate_histogram_distance(
    query_histogram,
    database_histogram
):
    """
    Calculate Euclidean distance between
    two normalized HSV histograms.
    """

    query_histogram = np.array(
        query_histogram,
        dtype=np.float64
    )

    database_histogram = np.array(
        database_histogram,
        dtype=np.float64
    )

    distance = np.linalg.norm(
        query_histogram - database_histogram
    )

    return distance


# ==================================================
# DOMINANT COLOR DISTANCE
# ==================================================

def calculate_color_distance(
    query_colors,
    database_colors
):
    """
    Calculate Euclidean distance between
    dominant-color feature vectors.
    """

    query_colors = np.array(
        query_colors,
        dtype=np.float64
    )

    database_colors = np.array(
        database_colors,
        dtype=np.float64
    )

    distance = np.linalg.norm(
        query_colors - database_colors
    )

    return distance


# ==================================================
# COMBINED SIMILARITY DISTANCE
# ==================================================

def calculate_similarity(
    histogram_distance,
    color_distance
):
    """
    Combine histogram and dominant-color
    distances using the project weights.

    Histogram      = 70%
    Dominant Color = 30%
    """

    similarity_distance = (
        0.70 * histogram_distance
        +
        0.30 * color_distance
    )

    return similarity_distance


# ==================================================
# RETRIEVE STORED FEATURES FROM ORACLE
# ==================================================

def get_database_features():

    connection = get_connection()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT
                i.image_id,
                i.filename,
                i.file_path,
                f.color1_h,
                f.color1_s,
                f.color1_v,
                f.color1_percentage,
                f.color2_h,
                f.color2_s,
                f.color2_v,
                f.color2_percentage,
                f.color3_h,
                f.color3_s,
                f.color3_v,
                f.color3_percentage,
                f.histogram_data
            FROM images i
            INNER JOIN image_features f
                ON i.image_id = f.image_id
            ORDER BY i.image_id
            """
        )

        rows = cursor.fetchall()

        # Convert Oracle CLOB into normal Python string
        processed_rows = []

        for row in rows:

            row = list(row)

            if row[15] is not None:
                row[15] = row[15].read()

            processed_rows.append(tuple(row))

        return processed_rows

    finally:

        cursor.close()
        connection.close()


# ==================================================
# FIND SIMILAR IMAGES
# ==================================================

def find_similar_images(
    query_image_path,
    top_n=5
):

    print()
    print("===================================")
    print("IMAGE RETRIEVAL")
    print("===================================")

    print("Query image:", query_image_path)

    # -----------------------------------------
    # Extract query features
    # -----------------------------------------

    query_histogram, query_colors = extract_query_features(
        query_image_path
    )

    print("Query histogram features:", len(query_histogram))
    print("Query color features:", len(query_colors))
    print("Query feature vector:", len(query_histogram) + len(query_colors))

    # -----------------------------------------
    # Get database features
    # -----------------------------------------

    rows = get_database_features()

    results = []

    # -----------------------------------------
    # Compare query with every database image
    # -----------------------------------------

    for row in rows:

        image_id = row[0]
        filename = row[1]
        file_path = row[2]

        # -------------------------------------
        # Dominant colors from database
        # -------------------------------------

        database_colors = np.array([
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),

            float(row[7]),
            float(row[8]),
            float(row[9]),
            float(row[10]),

            float(row[11]),
            float(row[12]),
            float(row[13]),
            float(row[14])
        ])

        # -------------------------------------
        # Histogram from database
        # -------------------------------------

        database_histogram = np.array(
            json.loads(row[15]),
            dtype=np.float64
        )

        # -------------------------------------
        # Calculate distances
        # -------------------------------------

        histogram_distance = calculate_histogram_distance(
            query_histogram,
            database_histogram
        )

        color_distance = calculate_color_distance(
            query_colors,
            database_colors
        )

        total_distance = calculate_similarity(
            histogram_distance,
            color_distance
        )

        # -------------------------------------
        # Store result
        # -------------------------------------

        results.append({
            "image_id": image_id,
            "filename": filename,
            "file_path": file_path,
            "histogram_distance": histogram_distance,
            "color_distance": color_distance,
            "similarity_distance": total_distance
        })

    # -----------------------------------------
    # Sort by distance
    # Lower distance = more similar
    # -----------------------------------------

    results.sort(
        key=lambda x: x["similarity_distance"]
    )

    return results[:top_n]


# ==================================================
# DISPLAY RESULTS
# ==================================================

def display_results(results):

    print()
    print("===================================")
    print("SIMILAR IMAGES")
    print("===================================")

    if not results:
        print("No images found.")
        return

    for rank, result in enumerate(results, start=1):

        print()
        print("Rank:", rank)
        print("Image ID:", result["image_id"])
        print("Filename:", result["filename"])
        print(
            "Histogram Distance:",
            round(result["histogram_distance"], 6)
        )
        print(
            "Color Distance:",
            round(result["color_distance"], 6)
        )
        print(
            "Total Distance:",
            round(result["similarity_distance"], 6)
        )


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    project_folder = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    image_path = os.path.join(
        project_folder,
        "images",
        "Screenshot 2026-06-21 144345.jpg"
    )

    results = find_similar_images(
        image_path,
        top_n=5
    )

    display_results(results)