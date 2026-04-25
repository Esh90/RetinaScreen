# RetinaScreen — Setup & Run Guide

## 1. Create Python 3.10 Virtual Environment

```bash
# Windows
python -m venv venv310
venv310\Scripts\activate

# macOS / Linux
python3.10 -m venv venv310
source venv310/bin/activate
```

If `python3.10` isn't your default, install it first:
```bash
# Ubuntu/Debian
sudo apt install python3.10 python3.10-venv

# macOS (Homebrew)
brew install python@3.10
```

## 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add Your Model Weights

Copy your trained model from Colab:
```
retinascreen/
└── weights/
    └── best_model.pth   ← place file here
```

Or load from HuggingFace Hub — set `HF_REPO` in `.streamlit/secrets.toml`:
```toml
HF_REPO = "your-username/retinascreen"
```

## 4. Run the App

```bash
streamlit run app.py
```

## 5. Deploy to Streamlit Cloud

```bash
# Push to GitHub first
git init
git add .
git commit -m "RetinaScreen v1.0"
git remote add origin https://github.com/your-username/retinascreen.git
git push -u origin main
```

Then go to https://share.streamlit.io → New app → connect your repo.

Add secrets in the Streamlit Cloud dashboard:
```
HF_REPO = "your-username/retinascreen"
```

## 6. Upload Model to HuggingFace (free hosting)

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="weights/best_model.pth",
    path_in_repo="weights/best_model.pth",
    repo_id="your-username/retinascreen",
    repo_type="model",
    token="YOUR_HF_TOKEN"
)
```

## Project Structure

```
retinascreen/
├── app.py                  ← Main Streamlit app
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── assets/
│   └── style.css
├── weights/
│   └── best_model.pth      ← Your trained weights
└── src/
    ├── __init__.py
    ├── model.py             ← Model architecture + loading
    ├── uncertainty.py       ← DUD / MC Dropout
    ├── gradcam.py           ← UM-GradCAM
    └── referral.py          ← Specialist finder (OSM)
```