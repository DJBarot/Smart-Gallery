
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

FEATURES = Path("features/clip")

gallery_embeddings = np.load(
    FEATURES / "gallery_embeddings.npy"
)

evaluation_embeddings = np.load(
    FEATURES / "evaluation_embeddings.npy"
)

with open(
    FEATURES / "gallery_metadata.csv",
    encoding="utf-8"
) as f:
    gallery_metadata = list(csv.DictReader(f))

with open(
    FEATURES / "evaluation_metadata.csv",
    encoding="utf-8"
) as f:
    evaluation_metadata = list(csv.DictReader(f))

# Choose an evaluation image
query_index = 0
top_k = 5

query_embedding = evaluation_embeddings[query_index]

# Both vectors are normalized, so dot product = cosine similarity
similarities = gallery_embeddings @ query_embedding

top_indices = np.argsort(similarities)[::-1][:top_k]

query = evaluation_metadata[query_index]

print("\nQUERY IMAGE")
print("Category:", query["category"])
print("Filename:", query["filename"])
print("Path:", query["image_path"])

print("\nTOP MATCHES")

fig, axes = plt.subplots(1, top_k + 1, figsize=(18, 4))

query_image = Image.open(query["image_path"]).convert("RGB")
axes[0].imshow(query_image)
axes[0].set_title("QUERY\n" + query["category"])
axes[0].axis("off")

for rank, index in enumerate(top_indices, start=1):

    item = gallery_metadata[index]
    score = float(similarities[index])

    print(
        f"{rank}. {item['filename']} | "
        f"Category: {item['category']} | "
        f"Similarity: {score:.4f}"
    )

    image = Image.open(item["image_path"]).convert("RGB")

    axes[rank].imshow(image)
    axes[rank].set_title(
        f"Rank {rank}\n"
        f"{item['category']}\n"
        f"Score: {score:.3f}"
    )
    axes[rank].axis("off")

plt.tight_layout()
plt.show()
