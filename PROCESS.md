# 🔁 Reproduction Guide - PROCESS.md

This guide explains how to reproduce the results, run the web app, and understand the workflow of the Smart Purchase Category Recommender project.

---

## 📦 Prerequisites

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
````

> Or run the project in [Google Colab](https://colab.research.google.com/drive/1i4ZGkAK_DpP7_7a8t1VeHtHjm6gAiL0r?usp=sharing) without local setup.

---

## 📂 Dataset Setup

The dataset is publicly available. It will be automatically downloaded using `gdown`.

If the automatic download fails, manually download the file from [Google Drive](https://drive.google.com/file/d/1XeVEFe9rW2KwWfVmMqVbR78quzKZlHrw/view?usp=sharing) and place it in the `data/` folder with the name:

```
Online_Shopping_Data.csv
```

---

## 🧠 Model Training (Optional)

If you want to retrain the model from scratch:

```bash
python src/train.py
```

> Trained model will be saved as `model.pth`.

If you're using the existing model (`model.pth`), skip this step.

---

## 🔍 Inference

To test the model's recommendation logic:

```bash
python src/inference.py
```

This will print sample predictions based on the embeddings.

---

## 🌐 Running the Web App

To launch the Streamlit application:

```bash
streamlit run src/app.py
```

You can interact with the app, choose a user ID, and view recommended product categories.

Or visit the **live version** here:
🔗 [Streamlit Demo](https://purchase-prediction-app-3fqjntw2sygip8mgd7scfa.streamlit.app/)

---

## 💻 Google Colab (No Setup Needed)

You can also open the Colab notebook directly here:
🔗 [Open in Colab](https://colab.research.google.com/drive/1i4ZGkAK_DpP7_7a8t1VeHtHjm6gAiL0r?usp=sharing)

---

## 📝 Notes

* Ensure the dataset file is named correctly and located in the `data/` folder
* If using Google Colab, you may need to mount Drive to upload model files
* Recommended for CPU usage; no GPU required unless retraining

---

✅ Once you follow the steps above, you will be able to:

* Run the full pipeline
* Launch the web app
* Reproduce the recommendation results

---
