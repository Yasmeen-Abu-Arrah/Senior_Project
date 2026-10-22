import os
import shutil
from huggingface_hub import snapshot_download

MODEL_DIR = "./qwen_model"

# Clean out old corrupted configuration states
if os.path.exists(MODEL_DIR):
    print("🧹 Cleaning old directory files...")
    shutil.rmtree(MODEL_DIR)

os.makedirs(MODEL_DIR, exist_ok=True)

print("⏳ Downloading full Qwen3-ASR-1.7B repository...")
try:
    # Pulling the native repository directly ensures 'rope_scaling' structural data loads properly
    snapshot_download(
        repo_id="Qwen/Qwen3-ASR-1.7B",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5"] # Ignores alternative framework weights to save space
    )
    print(f"\n🎉 Success! Model assets completely re-downloaded to: {os.path.abspath(MODEL_DIR)}")
except Exception as e:
    print(f"\n❌ Download failed. Error: {e}")
