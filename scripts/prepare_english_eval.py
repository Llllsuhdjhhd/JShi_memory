"""Download only the English encoder files, then verify local inference (no paid API)."""
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    path = snapshot_download("BAAI/bge-small-en-v1.5", allow_patterns=[
        "*.json", "*.txt", "model.safetensors", "1_Pooling/*", "*.md",
    ], max_workers=2)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(path, local_files_only=True, device="cpu")
    vectors = model.encode(["Caroline enjoys swimming.", "Represent this sentence for searching relevant passages: What sport does Caroline enjoy?"],
                           normalize_embeddings=True)
    print({"model": "BAAI/bge-small-en-v1.5", "dimension": int(vectors.shape[1]),
           "similarity": float(vectors[0] @ vectors[1])})
