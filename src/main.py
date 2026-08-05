import cv2
import numpy as np

def run_pipeline():
    print("--- Starting CG IP Baseline Test ---")
    print("\nGroup Members:")
    print("1) Deeparth Barot")
    print("2) Vasudev Kulkarni")
    print("3) Jyot Joshi")

    # Create a black canvas
    img = np.zeros((350, 700, 3), dtype=np.uint8)

    # Project Title
    cv2.putText(img, "CG IP Pipeline is Okay!", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Group Members
    cv2.putText(img, "Group Members:", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(img, "1) Deeparth Barot", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(img, "2) Vasudev Kulkarni", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(img, "3) Jyot Joshi", (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Create green edge image
    green_edges = np.zeros_like(img)
    green_edges[edges != 0] = (0, 255, 0)

    print("\nImage processed successfully.")

    # Display processed output
    cv2.imshow("Baseline Test Window", green_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline()