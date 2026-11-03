
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("\nLoading pretrained CLIP model...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model.eval()

print("\nCLIP loaded successfully!")
print("Embedding dimension:", model.config.projection_dim)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
