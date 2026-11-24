# Smart Image Gallery

A Computer Graphics & Image Processing (CGIP) project integrating Oracle Database, OpenCV (HSV color histograms & dominant colors), and Deep Learning (OpenAI CLIP embeddings) for content-based image retrieval and semantic text-to-image search.

---

## Project Structure

```
SmartImageGallery/
├── gui.py              # Main graphical desktop application (Tkinter)
├── main.py             # CLI application for image viewing, search, and indexing
├── .env                # Oracle database credentials and configuration
├── README.md           # Project documentation and guide
│
├── src/                # Core application package
│   ├── database.py             # Oracle DB connection manager
│   ├── feature_extraction.py   # HSV histogram and dominant color extraction (OpenCV)
│   ├── feature_visualization.py# Visualization utilities for color distributions
│   ├── image_manager.py        # Oracle DB image indexing and metadata pipeline
│   ├── similarity.py           # Color-based similarity and distance calculations
│   ├── embedding_manager.py    # CLIP embedding store loader and validator
│   ├── sync_clip_embeddings.py # Incremental CLIP embedding synchronization
│   └── text_search.py          # Semantic text-to-image & image-to-image CLIP search
│
├── images/             # Active image gallery collection
│
├── features/           # Persistent extracted feature vectors and embeddings
│   └── clip/
│       ├── gallery_embeddings.npy       # Normalized CLIP image embeddings (512-d)
│       ├── gallery_metadata.csv         # Metadata mapping embeddings to images
│       ├── gallery_color_histograms.npy # Persistent color histograms
│       ├── evaluation_embeddings.npy    # Precomputed evaluation query embeddings
│       └── evaluation_metadata.csv      # Evaluation queries ground truth
│
├── evaluation/         # Experimental evaluation scripts and benchmark datasets
│   ├── evaluate_clip.py                 # Precision@K, Recall@K, MRR & search latency
│   ├── evaluate_hybrid_search.py        # Hybrid CLIP + Color evaluation
│   ├── evaluate_text_search.py          # Semantic text query evaluation
│   ├── generate_color_features.py       # Benchmark color feature generator
│   ├── hybrid_evaluation_details.csv    # Query-level hybrid retrieval results
│   ├── hybrid_evaluation_summary.csv    # Aggregate hybrid performance metrics
│   └── text_search_evaluation.csv       # Aggregate text search performance metrics
│
└── tests/              # Verification and diagnostic test scripts
    ├── check_db.py                      # Verifies Oracle DB connection and tables
    ├── check_clip_mapping.py            # Validates CLIP embeddings alignment with DB
    ├── test_clip.py                     # Quick sanity check for PyTorch & CLIP model
    ├── test_clip_image_search.py        # Tests image-based CLIP similarity queries
    ├── test_image_search.py             # Visual test for query-by-image search
    ├── test_hybrid_search.py            # Tests weighted hybrid scoring
    └── text_image_search.py             # Interactive text query testing script
```

---

## How to Run

### 1. Graphical User Interface (GUI)
Launch the full desktop application:
```bash
python gui.py
```

### 2. Command Line Interface (CLI)
Run the terminal-based menu:
```bash
python main.py
```

### 3. Re-indexing Images
To index new images and synchronize CLIP embeddings from the CLI:
```bash
python -m src.image_manager
python -m src.sync_clip_embeddings
```
*(Or click **Index Images** directly inside the GUI)*

### 4. Running Tests and Diagnostics
```bash
python tests/check_db.py
python tests/test_clip.py
```

### 5. Running Evaluations
```bash
python evaluation/evaluate_clip.py
```
