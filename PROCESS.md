# 🔁 Reproduction Guide - PROCESS.md

This guide walks through how to reproduce results, train the model, and run the web application for the Smart Purchase Category Recommender project.

---

## 📦 Prerequisites

* Python 3.10+
* Install required packages:

```bash
pip install -r requirements.txt
```

> No setup? Use [Google Colab](https://colab.research.google.com/drive/1i4ZGkAK_DpP7_7a8t1VeHtHjm6gAiL0r?usp=sharing).

---

## 📂 Dataset Setup

The dataset will be auto-downloaded using `gdown`:

```python
!gdown --id 1XeVEFe9rW2KwWfVmMqVbR78quzKZlHrw --output data/Online_Shopping_Data.csv
```

Alternatively, manually download it from [Google Drive](https://drive.google.com/file/d/1XeVEFe9rW2KwWfVmMqVbR78quzKZlHrw/view?usp=sharing) and save it in the `data/` folder.

---

## 🧠 Model Training (Optional)

To retrain the model from scratch:

```bash
python src/train.py
```

> This generates a new `model.pth` and updated embedding files.

Skip this if you are using the provided `model.pth`.

---

## 🔍 Inference

To test predictions from embeddings:

```bash
python src/inference.py
```

Sample recommendations will be printed in the terminal.

---

## 🌐 Run the Streamlit App

Launch the web application:

```bash
streamlit run src/app.py
```

Interactively explore:

* Customer profiles
* Category recommendations

---

## 💻 Google Colab Option

Run the notebook in-browser without setup:

[Open in Colab](https://colab.research.google.com/drive/1i4ZGkAK_DpP7_7a8t1VeHtHjm6gAiL0r?usp=sharing)

---

## 📃 Summary of Artifacts

* `model.pth`: Trained PyTorch model
* `user_embeddings.pkl`: Serialized user vectors
* `category_embeddings.pkl`: Serialized category vectors
* `Online_Shopping_Data.csv`: Source data
* `src/`: All logic for training, inference, and app

---

## ✨ Notes

* The model is deterministic with fixed random seeds:

```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

* No GPU required for inference or web app
* All SBERT encodings are precomputed for speed

---

Once completed, you can reproduce the results, re-train the model, and deploy the app with confidence.
