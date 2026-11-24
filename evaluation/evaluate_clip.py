
from pathlib import Path
import csv
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES = PROJECT_ROOT / "features" / "clip"

gallery = np.load(FEATURES / "gallery_embeddings.npy")
queries = np.load(FEATURES / "evaluation_embeddings.npy")

with open(FEATURES / "gallery_metadata.csv", encoding="utf-8") as f:
    gallery_meta = list(csv.DictReader(f))

with open(FEATURES / "evaluation_metadata.csv", encoding="utf-8") as f:
    query_meta = list(csv.DictReader(f))

assert len(gallery) == len(gallery_meta)
assert len(queries) == len(query_meta)

# Verify gallery category counts
gallery_categories = np.array(
    [item["category"] for item in gallery_meta]
)

print("Gallery category counts:")

for category in sorted(set(gallery_categories)):
    count = np.sum(gallery_categories == category)
    print(f"{category}: {count}")

precision5 = []
precision10 = []
recall5 = []
recall10 = []
reciprocal_ranks = []
hit_rate5 = []
search_times = []

category_results = {}

for i, query in enumerate(queries):

    start = time.perf_counter()

    scores = gallery @ query
    ranked_indices = np.argsort(scores)[::-1]

    search_times.append(time.perf_counter() - start)

    query_category = query_meta[i]["category"]

    relevant_total = np.sum(
        gallery_categories == query_category
    )

    relevant = (
        gallery_categories[ranked_indices] == query_category
    )

    relevant_top5 = int(np.sum(relevant[:5]))
    relevant_top10 = int(np.sum(relevant[:10]))

    p5 = relevant_top5 / 5
    p10 = relevant_top10 / 10

    r5 = relevant_top5 / relevant_total
    r10 = relevant_top10 / relevant_total

    positions = np.where(relevant)[0]

    mrr = (
        1 / (positions[0] + 1)
        if len(positions) else 0
    )

    precision5.append(p5)
    precision10.append(p10)
    recall5.append(r5)
    recall10.append(r10)
    reciprocal_ranks.append(mrr)
    hit_rate5.append(relevant_top5 > 0)

    if query_category not in category_results:
        category_results[query_category] = []

    category_results[query_category].append({
        "precision5": p5,
        "precision10": p10,
        "recall5": r5,
        "recall10": r10,
        "mrr": mrr
    })

print("\n========== CORRECTED CLIP EVALUATION ==========")

print("Queries:", len(queries))
print("Gallery images:", len(gallery))

print(f"\nPrecision@5:  {np.mean(precision5) * 100:.2f}%")
print(f"Precision@10: {np.mean(precision10) * 100:.2f}%")

print(f"Recall@5:     {np.mean(recall5) * 100:.4f}%")
print(f"Recall@10:    {np.mean(recall10) * 100:.4f}%")

print(f"MRR:          {np.mean(reciprocal_ranks):.4f}")
print(f"HitRate@5:    {np.mean(hit_rate5) * 100:.2f}%")

print(
    f"Average search time: "
    f"{np.mean(search_times) * 1000:.3f} ms"
)

print("\n========== CATEGORY-WISE RESULTS ==========")

for category, results in sorted(category_results.items()):

    print(f"\n{category.upper()}")

    for metric in [
        "precision5", "precision10",
        "recall5", "recall10", "mrr"
    ]:
        value = np.mean([x[metric] for x in results])

        if metric.startswith(("precision", "recall")):
            print(f"{metric}: {value * 100:.2f}%")
        else:
            print(f"{metric}: {value:.4f}")
