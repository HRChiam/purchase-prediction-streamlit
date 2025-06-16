# 🛍️ Smart Purchase Category Recommender

An AI-powered shopping assistant that recommends product categories to users based on their shopping behavior. Built using **PyTorch**, **SBERT**, and **Streamlit**.

---

## 📦 Features

- ✅ Select a real customer from the dataset  
- ✅ View summarized shopping behavior  
- ✅ Get top-N personalized category recommendations  
- ✅ Fast inference with SBERT embeddings  
- ✅ Easy-to-use Streamlit interface  

---

## 📊 Project Overview

This project simulates a personalized e-commerce assistant that recommends product categories based on user purchasing behavior. It leverages:

- **Contrastive learning** to train on user-category similarity
- **SBERT embeddings** to encode user profiles and product categories
- **Streamlit** for interactive visualization and inference

---

## 📁 Project Structure

```text
purchase-prediction-streamlit/
├── data/                         # Dataset (manual or auto-downloaded)
├── saved/                        # Folder to store models and embeddings
├── src/                          # Source code
│   ├── __init__.py
│   ├── app.py                    # Streamlit app entry point
│   ├── inference.py              # Prediction and embedding matching
│   ├── model_utils.py            # Contrastive model setup
│   ├── preprocess_utils.py       # Preprocessing functions
│   └── train.py                  # Training script
├── model.pth                     # Trained model checkpoint
├── requirements.txt              # Project dependencies
├── README.md                     # This file
└── .gitignore                    # Ignore logs, models, etc.
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HRChiam/purchase-prediction-streamlit.git
cd purchase-prediction-streamlit
```

### 2. Install Dependencies

Make sure you are using **Python 3.10+**.

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Auto-download via:

```python
!gdown --id 1XeVEFe9rW2KwWfVmMqVbR78quzKZlHrw --output data/Online_Shopping_Data.csv
```

Or manually place `Online_Shopping_Data.csv` into the `data/` directory.

---

### 4. Run the Streamlit App

```bash
streamlit run src/app.py
```

---

## 🧠 Model Details

- **Model Type**: Siamese Contrastive Learning Model (PyTorch)
- **Text Encoder**: [SBERT (Sentence-BERT)](https://www.sbert.net/)
- **Learning Objective**: Learn embeddings that bring similar users and categories closer in vector space.

### 📦 Saved Artifacts

- `model.pth` – Trained PyTorch model
- `user_embeddings.pkl` – Precomputed user vectors
- `category_embeddings.pkl` – Precomputed category vectors

---

## 🧪 Reproducibility

Ensure consistent results by fixing random seeds:

```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

Run training and inference reproducibly using:

- `src/train.py`
- `src/inference.py`

---

## 🧑‍💻 Contributors

Developed by:

- Yap Yu Hang  
- Tham Wing Shan  
- Tan Wei Ren  
- Chiam Huai Ren  
- Liu Yi Xian  

For academic and demo purposes.
