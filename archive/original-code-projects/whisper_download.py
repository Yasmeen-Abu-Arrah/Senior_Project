import os
import whisper

# 1. Define your local storage folder inside the project
MODELS_DIR = "./whisper_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 2. Get the official list of all available models from Whisper
# This automatically includes: tiny, base, small, medium, large, turbo, and .en variants
all_models = list(whisper._MODELS.keys())

print(f"🚀 Found {len(all_models)} Whisper models to download.")
print(f"📂 Saving all models locally to: {os.path.abspath(MODELS_DIR)}\n")

# 3. Loop and download each model
for model_name in all_models:
    print(f"⏳ Downloading [{model_name}]...")
    try:
        # download_root forces Whisper to save the .pt file into your specified folder
        whisper.load_model(model_name, download_root=MODELS_DIR)
        print(f"✅ [{model_name}] downloaded successfully.\n")
    except Exception as e:
        print(f"❌ Failed to download [{model_name}]. Error: {e}\n")

print("🎉 All available Whisper models have been downloaded and are ready for offline use!")
