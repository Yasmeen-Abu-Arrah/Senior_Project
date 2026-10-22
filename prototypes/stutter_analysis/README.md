# Stutter Analysis

Live microphone and file-based stutter detection using a custom wav2vec2
classifier. Classes: fluent, repetition, prolongation, block.

Run the demo from the repository root:

```powershell
python prototypes\stutter_analysis\app.py
```

Needs `models/stutter_classifier/` (`model.pt`, `feature_extractor/`,
`metadata.json`). This trained artifact is not publicly downloadable.

Training and evaluation are handled by `stutter_model_trainer.py`. Dataset
commands in its docstring use paths relative to the repository root:

```powershell
python prototypes\stutter_analysis\stutter_model_trainer.py --mode evaluate --data_dir data\stuttered --output_model models\stutter_classifier
```
