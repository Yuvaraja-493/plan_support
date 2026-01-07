# Plan Prediction API

Production-ready FastAPI service for predicting clean plan names, plan types, and LOB.

## 📁 Project Structure

```
├── api/                 # FastAPI application
│   └── main.py
├── dataset/             # Training data
│   └── Iris Plan_Mapping Data_Top 100.xlsx
├── models/              # Trained models
│   ├── le_*.joblib (3 files)
│   ├── model_*.joblib (3 files)
│   └── sentence_transformer_model_name.txt
├── requirements.txt     # Dependencies
├── train.py             # Training script
├── start_api.bat        # API startup script
└── README.md            # This file
```

## 🚀 Quick Start

**Start API:**
```cmd
start_api.bat
```

**Access:** http://localhost:8000/docs

## 🔄 Retrain Models

```powershell
python train.py
```

Models will be saved to `models/` folder.

## 📝 API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "payer_name": "AARP Medicare Complete",
        "dirty_plan_name": "LPPO-AARP MEDICARE ADVANTAGE..."
    }
)
print(response.json())
```

## 📊 Response

```json
{
  "clean_plan_name": "...",
  "plan_type": "PPO",
  "line_of_business": "Medicare"
}
```

## 📂 Folders

- **api/** - FastAPI server code
- **dataset/** - Training data files
- **models/** - Trained model files (.joblib)
- **venv/** - Python virtual environment
