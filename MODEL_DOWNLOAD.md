# Downloading Model and Dataset

## Quick Setup

The trained model and dataset are not included in this repository due to their size. You have two options:

### Option 1: Automatic Download (Recommended)

Run the download helper script:

```bash
# Download model only (required, ~13MB)
python download_assets.py

# Download model + dataset (optional, ~85MB total)
python download_assets.py --dataset
```

### Option 2: Manual Download

1. **Download the trained model** (~13 MB):
   - [cnn8grps_rad1_model.h5](https://github.com/souvik001122/Sign2TextSpeech/releases/download/v1.0/cnn8grps_rad1_model.h5)
   - Place it in the project root directory

2. **Download the dataset** (optional, ~72 MB):
   - [AtoZ_3.1.zip](https://github.com/souvik001122/Sign2TextSpeech/releases/download/v1.0/AtoZ_3.1.zip)
   - Extract to `AtoZ_3.1/` folder in project root
   - Required only for training or data collection

## File Structure After Download

```
ml_project/
├── cnn8grps_rad1_model.h5    # Trained model (required)
├── AtoZ_3.1/                  # Dataset (optional)
│   ├── A/
│   ├── B/
│   └── ...
├── final_pred.py              # Main application
└── ...
```

## Cloud Storage Alternatives

If GitHub Releases are not available, you can also download from:

- **Google Drive**: [Model & Dataset](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)
- **Dropbox**: [Shared Folder](https://www.dropbox.com/sh/YOUR_SHARE_LINK)
- **OneDrive**: [Assets](https://onedrive.live.com/YOUR_LINK)

*(Update these links after uploading to your preferred cloud storage)*

## Verifying Downloads

After downloading, verify the files:

```bash
# Check if model exists
python -c "import os; print('✓ Model found' if os.path.exists('cnn8grps_rad1_model.h5') else '✗ Model missing')"

# Check if dataset exists
python -c "import os; print('✓ Dataset found' if os.path.exists('AtoZ_3.1') else '✗ Dataset missing (optional)')"
```

## Running the Application

Once the model is downloaded, you can run:

```bash
python final_pred.py
```

For more details, see the main [README.md](README.md).
