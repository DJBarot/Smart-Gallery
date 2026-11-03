# Smart Image Gallery

**An AI-powered desktop application for content-based image retrieval using natural language and visual similarity.**

Smart Image Gallery helps users find images by describing what they are looking for or by supplying a query image. The main semantic retrieval workflow uses **OpenAI CLIP (ViT-B/32)** to encode text and images into a shared embedding space, then ranks gallery items using cosine similarity.

The project also includes an HSV-based image-processing and similarity workflow backed by Oracle Database.

## Features

- **Search by Text:** Enter a natural-language description (for example, `a dense green forest with tall trees`) and retrieve matching gallery images.
- **Search by Image:** Select an image and retrieve visually/semantically similar gallery items using CLIP image embeddings.
- **Image Gallery:** Browse indexed image records stored in Oracle Database.
- **Image Import:** Import images into the application's local image collection.
- **Image Indexing:** Extract image features, store image metadata/features in Oracle, and synchronize CLIP embeddings.
- **Image Processing:** Apply image preprocessing and inspect colour-based features.
- **3D Gallery Showcase:** `immersive_gallery.html` is a separate browser-based visual showcase.

## How it works

### CLIP semantic retrieval

1. **Input:** The user enters a text prompt or selects a query image.
2. **Encoding:** CLIP ViT-B/32 converts the query into a 512-dimensional feature vector.
3. **Normalization:** The query and stored gallery embeddings are L2-normalized.
4. **Similarity:** The application computes cosine similarity. Since the vectors are normalized, this is a dot product:

   ```text
   similarity(query, image) = query_embedding · image_embedding
   ```

5. **Ranking:** Gallery images are sorted by similarity and the Top-K results are displayed.

Gallery embeddings are stored in `features/clip/` and loaded from the local embedding store. The application expects the gallery metadata to contain the image ID, filename and image path.

### HSV image-processing workflow

The image-processing modules also provide a colour-based workflow:

- Resize images to 256 × 256.
- Apply a 5 × 5 Gaussian blur.
- Convert the image from BGR to HSV.
- Extract a normalized 16 × 4 × 4 HSV histogram (256 histogram values).
- Use K-Means (`k=3`) to extract dominant colours.
- Compare colour features using weighted distances (70% histogram distance and 30% dominant-colour distance).

The CLIP and HSV approaches are evaluated separately and through the tested hybrid score combinations. The tested HSV fusion weights did not outperform CLIP-only retrieval in the supplied evaluation.

## Technology stack

- Python
- OpenAI CLIP — `openai/clip-vit-base-patch32`
- PyTorch and Hugging Face Transformers
- OpenCV, Pillow and NumPy
- scikit-learn, pandas, Matplotlib and tqdm
- Tkinter desktop GUI
- Oracle Database with `python-oracledb`
- HTML, CSS and JavaScript for the standalone gallery showcase

## Repository structure

The main application files should be placed at the repository root, with the `src/` package kept alongside them:

```text
Smart-Gallery/
├── gui.py
├── main.py
├── immersive_gallery.html
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── embedding_manager.py
│   ├── feature_extraction.py
│   ├── feature_visualization.py
│   ├── image_manager.py
│   ├── similarity.py
│   ├── sync_clip_embeddings.py
│   └── text_search.py
├── images/                 # Local image collection; not committed by default
└── features/
    └── clip/               # Generated embeddings and metadata; not committed by default
```

Keep the complete `src/` directory. `gui.py` and `main.py` import modules from it and will not run correctly if the package is omitted.

## Requirements

- Python 3.10 or newer is recommended.
- Oracle Database must be reachable from the computer running the application.
- A working internet connection is needed the first time CLIP is loaded so Transformers can download the model weights, unless the model is already cached.
- Tkinter is included with many standard Python installations. On some Linux distributions it must be installed separately (for example, `python3-tk`).

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DJBarot/Smart-Gallery.git
cd Smart-Gallery
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can vary depending on whether you want CPU-only execution or CUDA/GPU acceleration. If the default `torch` installation does not match your machine, install the appropriate build using the official PyTorch installation selector, then run the requirements installation again if needed.

## Oracle Database configuration

The application reads the database password from the `ORACLE_PASSWORD` environment variable. **Do not put database passwords in source code or commit them to GitHub.**

The current `src/database.py` configuration uses these defaults:

| Setting | Current value |
|---|---|
| Username | `SYSTEM` |
| Host | `djb` |
| Port | `1521` |
| Service name | `free` |
| Password | Read from `ORACLE_PASSWORD` |

Update the host, username, port and service name in `src/database.py` to match your Oracle installation before running the application. For a shared or deployed environment, use a dedicated least-privilege database account instead of the Oracle `SYSTEM` account.

**Set the password for the current terminal session:**

Windows PowerShell:

```powershell
$env:ORACLE_PASSWORD = "your_oracle_password"
```

macOS / Linux:

```bash
export ORACLE_PASSWORD="your_oracle_password"
```

The application code expects the Oracle tables used by `src/image_manager.py`, `src/similarity.py` and the GUI, including `IMAGES` and `IMAGE_FEATURES`. The supplied ZIP does **not** contain a database schema/setup SQL script. Create the required tables in your Oracle schema before using database-backed features.

You can test the Oracle connection from the project root with:

```bash
python -m src.database
```

## Run the application

Run commands from the repository root (the folder containing `gui.py`, `main.py` and `src/`).

### Desktop GUI

```bash
python gui.py
```

The GUI provides the gallery, text search, image search, image processing, image import and indexing screens.

### Command-line interface

```bash
python main.py
```

The CLI provides options to view indexed images, index images, search by image, search by text and exit.

### 3D gallery showcase

Open `immersive_gallery.html` in a modern web browser. It is a standalone visual showcase and is separate from the Python desktop application's Oracle/CLIP search backend.

## Image indexing and CLIP embeddings

For the main GUI workflow:

1. Configure Oracle and make sure the required tables exist.
2. Open the GUI using `python gui.py`.
3. Use **Import Images** to add images to the local collection.
4. Use **Index Images** to run the Oracle/HSV indexing step and synchronize CLIP embeddings.
5. Search using text or an image.

The indexing workflow invokes `src.image_manager` and `src.sync_clip_embeddings`. The generated CLIP store is written under `features/clip/`.

The separate `generate_embeddings.py` and evaluation scripts are research/experiment utilities. Some of those scripts contain machine-specific dataset paths; update those paths to your own gallery and evaluation folders before running them. Do not assume those scripts use the same metadata format as the live GUI embedding store.

## Evaluation results

The supplied evaluation files report category-based retrieval precision. These values are **not classification accuracy**.

### Text-to-image retrieval

Six natural-language prompts (one for each gallery category) were evaluated:

| Metric | Result |
|---|---:|
| Precision@5 | **90.00%** |
| Precision@10 | **86.67%** |

### Image-to-image retrieval

Sixty query images were evaluated against the indexed gallery:

| Method | Precision@5 | Precision@10 |
|---|---:|---:|
| CLIP only | **88.67%** | **87.17%** |
| CLIP 90% + HSV 10% | 87.00% | 86.00% |
| CLIP 80% + HSV 20% | 86.00% | 84.83% |
| CLIP 70% + HSV 30% | 83.00% | 82.33% |
| CLIP 60% + HSV 40% | 80.33% | 76.83% |
| CLIP 50% + HSV 50% | 75.00% | 71.83% |

In these experiments, CLIP-only retrieval performed better than the tested CLIP + HSV fusion configurations. Results are based on category-label relevance in the supplied evaluation set; they do not establish relevance for every possible query or dataset.

The evaluation scripts and result files are included in the project archive (for example, `evaluate_text_search.py`, `evaluate_clip.py`, `evaluate_hybrid_search.py`, `text_search_evaluation.csv` and the hybrid evaluation CSV files).

## Important notes for contributors

- Keep `src/` in the repository; it contains the modules imported by the application.
- Do not commit `.env` files, Oracle passwords, local database credentials or private image collections.
- The `.gitignore` excludes generated embedding arrays and local datasets. Each contributor should create or synchronize their own local embedding store.
- Large datasets and model weights should be distributed separately or through an appropriate data-storage service, not committed as ordinary Git files.
- Before publishing, verify that you have permission to share all images and other assets in the repository.

## Project

**Smart Image Gallery** — content-based image retrieval using visual and textual understanding.
