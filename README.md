# 🏭️ Purchase Category Recommender

An AI-powered shopping assistant that recommends product categories based on users' shopping behavior. Built using **PyTorch**, **SBERT**, and **Streamlit**.

---

## 📦 Features

* Select a real customer from the dataset
* View summarized shopping behavior
* Get top-N personalized category recommendations
* Fast inference with SBERT embeddings
* Simple and interactive Streamlit interface

---

## 📊 Project Overview

This project simulates a personalized e-commerce assistant that recommends product categories based on user purchasing behavior. It uses:

* **Contrastive learning** to learn similarities between users and product categories
* **SBERT embeddings** to encode user and product profiles into vector space
* **Streamlit** for interactive app deployment

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
├── README.md                     # Project overview and usage
├── PROCESS.md                    # Step-by-step reproduction guide
└── .gitignore                    # Files to exclude from Git
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HRChiam/purchase-prediction-streamlit.git
cd purchase-prediction-streamlit
```

### 2. Install Dependencies

Ensure Python 3.10+ is installed.

```bash
pip install -r requirements.txt
```

### 3. Dataset Setup

Automatically download dataset:

```python
!gdown --id 1XeVEFe9rW2KwWfVmMqVbR78quzKZlHrw --output data/Online_Shopping_Data.csv
```

If download fails, manually place the file in `data/` folder as `Online_Shopping_Data.csv`.

### 4. Launch the Web App

```bash
streamlit run src/app.py
```

---

## 🧠 Model Details

* **Model**: Siamese Contrastive Learning
* **Embeddings**: SBERT (Sentence-BERT)
* **Objective**: Learn a shared embedding space for users and product categories

### 📦 Artifacts

* `model.pth`: Trained model checkpoint
* `user_embeddings.pkl`: Precomputed user vectors
* `category_embeddings.pkl`: Precomputed category vectors

---

## 🔪 Reproducibility

Set random seeds to ensure reproducible results:

```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

To reproduce the full pipeline, follow [PROCESS.md](./PROCESS.md).

---

## 👨‍💼 Contributors

Created by:

* Yap Yu Hang
* Tham Wing Shan
* Tan Wei Ren
* Chiam Huai Ren
* Liu Yi Xian

For academic and demonstration use.

---
