import os
import urllib.request

# Create a fonts directory if it doesn't exist
FONT_DIR = "./fonts"
os.makedirs(FONT_DIR, exist_ok=True)

# URL to the official stable Amiri Regular font from Google Fonts upstream
AMIRI_URL = "https://github.com"
FONT_PATH = os.path.join(FONT_DIR, "Amiri-Regular.ttf")

print("⏳ Downloading Amiri Font for offline usage...")
try:
    urllib.request.urlretrieve(AMIRI_URL, FONT_PATH)
    print(f"✅ Amiri font downloaded successfully and saved locally to: {FONT_PATH}")
except Exception as e:
    print(f"❌ Failed to download font: {e}")
