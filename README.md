# 🛍️ Smart Purchase Category Recommender

This is an AI-powered web app that recommends product categories to users based on their shopping behavior. It uses contrastive learning and SBERT embeddings to match user patterns with category vectors.

---

## 📦 Features

- Select a customer from real data  
- View behavioral summary  
- Get top-N recommended categories  
- Powered by PyTorch and Streamlit  

---

## 📊 Project Overview

This project simulates a personalized shopping assistant that recommends product categories based on user behavior. It uses **contrastive learning** with **SBERT embeddings** to understand purchase patterns and match them with similar categories.

---

## 📁 Project Structure

```text
purchase-prediction-streamlit/
├── data/                             # Dataset folder (optional/manual download)
├── saved/                            # Folder to store outputs or model files
├── src/                              # Source code
│   ├── _init_.py                   # Init file for module
│   ├── app.py                        # Streamlit app interface
│   ├── inference.py                  # Inference functions for prediction
│   ├── model_utils.py                # Model architecture and utilities
│   ├── preprocess_utils.py           # Data cleaning and preprocessing functions
│   └── train.py                      # Model training script
├── model.pth                         # Trained PyTorch model file
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore rules

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HRChiam/purchase-prediction-streamlit.git
cd purchase-prediction-streamlit
````

### 2. Install Dependencies

Make sure you are using **Python 3.10+**:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run src/deployment/app.py
```

---

## 📂 Dataset

* **Source**: Public Kaggle dataset
* **Auto-download** in code:

```python
!gdown --id 1XeVEFe9rW2KwWfVmMqVbR78quzKZlHrw --output Online_Shopping_Data.csv
```

* If this fails, you may manually download and place the file in the `data/` directory.

---

## 🧠 Model Details

* **Model**: Contrastive learning with PyTorch
* **Embedding**: Sentence-BERT (SBERT)
* **Fixed seeds for reproducibility**:

```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### 📦 Artifacts

* `contrastive_model.pt` – Saved trained model
* `user_embeddings.pkl`, `category_embeddings.pkl` – Precomputed vector files

---

## 🧪 Reproducibility

* All steps are reproducible via:

  * `PROCESS.md` instructions
  * `run_all.ipynb` or `main.py` (if available)

### 🔖 Git Version Tags

* `v1.0`: Initial model
* `v1.1`: Deployed version

---

## 🙌 Credits

Developed by **Yap Yu Hang**, **Tham Wing Shan**, **Tan Wei Ren**, **Chiam Huai Ren**, and **Liu Yi Xian** for educational and demonstration purposes.

```

---
