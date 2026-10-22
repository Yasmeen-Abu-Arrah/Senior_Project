req = set()
mapping = {
 'gradio': 'gradio>=4.44.0',
 'whisper': 'openai-whisper==20240930',
 'torch': 'torch>=2.4.0',
 'torchaudio': 'torchaudio>=2.4.0',
 'torchvision': 'torchvision>=0.19.0',
 'transformers': 'transformers>=4.45.0',
 'cv2': 'opencv-python>=4.10.0.84',
 'mediapipe': 'mediapipe>=0.10.14',
 'soundfile': 'soundfile>=0.12.1',
 'numpy': 'numpy>=1.26.0',
 'huggingface_hub': 'huggingface-hub>=0.25.0',
 'f5_tts': 'f5-tts>=1.1.0',
 'accelerate': 'accelerate>=1.0.0',
 'datasets': 'datasets>=3.0.0',
 'tqdm': 'tqdm>=4.66.0',
 'cached_path': 'cached-path>=1.6.0',
 'hydra': 'hydra-core>=1.3.0',
 'omegaconf': 'omegaconf>=2.3.0',
 'tomli': 'tomli>=2.0.1',
 'unidecode': 'Unidecode>=1.3.8',
 'click': 'click>=8.1.7',
 'psutil': 'psutil>=6.0.0',
 'spaces': 'spaces>=0.30.0',
 'editdistance': 'editdistance>=0.8.1',
 'regex': 'regex>=2024.9.11',
 'elevenlabs': 'elevenlabs>=1.50.0',
}
import re
with open(r'D:\Projects\Communication Disorders\imports_summary.txt', encoding='utf-8') as f:
    content = f.read()
for m in re.findall(r'^(?:import|from)\s+([a-zA-Z0-9_\.]+)', content, re.M):
    top = m.split('.')[0]
    if top in mapping:
        req.add(mapping[top])
# defaults
req.add('gradio>=4.44.0')
req.add('openai-whisper==20240930')
req.add('torch>=2.4.0')
req.add('torchaudio>=2.4.0')
req.add('transformers>=4.45.0')
req.add('huggingface-hub>=0.25.0')
req.add('soundfile>=0.12.1')
req.add('opencv-python>=4.10.0.84')
req.add('mediapipe>=0.10.14')
req.add('numpy>=1.26.0')
req_list = sorted(list(req))
req_out = r'D:\Projects\Communication Disorders\requirements-generated.txt'
with open(req_out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(req_list) + '\n')
print(f'Wrote {req_out} ({len(req_list)} packages)')
