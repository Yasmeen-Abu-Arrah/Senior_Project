import os
from huggingface_hub import snapshot_download

MODEL_DIR = "./qwen_text_model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("⏳ Downloading Qwen2.5-3B-Instruct for offline story generation...")
try:
    snapshot_download(
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.gguf"]
    )
    print(f"\n🎉 Success! Model saved to: {os.path.abspath(MODEL_DIR)}")
except Exception as e:
    print(f"\n❌ Download failed. Error: {e}")
