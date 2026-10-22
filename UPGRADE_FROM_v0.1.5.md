# Resume an interrupted v0.1.5 Colab run

This release fixes the three v0.1.5 blockers: the Carinthia JPEG hash, the
MIIC configuration-key mismatch, and the MIIC duplicate-component split. The
last fix is intentionally conservative: it uses decoded-pixel identity rather
than pHash+SSIM, quarantines the 182 inpainting pairs that share source images
with the sealed final test, and performs the remaining pair split by exact
duplicate component.

Your verified MIIC archives, extracted datasets, and the completed Carinthia
preparation are reusable. Do not delete or redownload them.

In the same Colab runtime, run this one-time upgrade cell after placing
`adaptive-sem-inspection-v0.1.6-colab-python312.zip` in MyDrive:

```python
from pathlib import Path
import os
import zipfile

ZIP_ON_DRIVE = Path("/content/drive/MyDrive/adaptive-sem-inspection-v0.1.6-colab-python312.zip")
PROJECT = Path("/content/adaptive-sem-source-v0.1.6")
RUNTIME_ROOT = Path("/content/drive/MyDrive/adaptive-sem-inspection-runtime-v0.1.5")
MIIC_MANUAL_BUNDLE = Path("/content/drive/MyDrive/MIIC_verified_files")

if not ZIP_ON_DRIVE.is_file():
    raise FileNotFoundError(f"Upload the corrected ZIP first: {ZIP_ON_DRIVE}")
if not (PROJECT / "scripts" / "run_colab_pipeline.py").is_file():
    PROJECT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_ON_DRIVE) as archive:
        archive.extractall(PROJECT)

os.chdir(PROJECT)
print("Project:", PROJECT)
print("Reusing runtime:", RUNTIME_ROOT)
```

Then rerun the cell that defines `run_stage` so it uses the new `PROJECT`
value. Keep the existing Python 3.12 runner and current 7-Zip setup. Finally,
run this once:

```python
run_stage(
    "development",
    "--install",
    "--accept-miic-license",
    "--miic-manual-bundle",
    str(MIIC_MANUAL_BUNDLE),
)
```

The launcher validates and reuses completed data steps, rebuilds the split
manifest with the corrected rules, and then continues through features,
calibration, router training, and policy calibration. Do not run the final
stage until development completes successfully.
