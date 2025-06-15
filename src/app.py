import streamlit as st
import os
import torch
import numpy as np
import pandas as pd
from inference import predict_top_k
from model_utils import ContrastiveModel
from preprocess_utils import load_and_preprocess_data

# Page configuration
st.set_page_config(page_title="🛒 Smart Category Recommender", layout="centered")

# Title
st.markdown("""
    <h1 style='text-align: center; color: #FF6F61;'>🛍️ Purchase Category Recommender</h1>
    <hr style='border-top: 1px solid #ccc;'>
""", unsafe_allow_html=True)

# Load Data and Model
with st.spinner("🔄 Loading data and model..."):
    csv_path = os.path.join("data", "Online_Shopping_Data.csv")
    user_df, cat_df, cat_combined, cat_lookup_array, user_features, le_category = load_and_preprocess_data(csv_path, use_saved_embeddings=True)

    model = ContrastiveModel(input_user=len(user_features), input_cat=cat_combined.shape[1], emb_dim=64)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()

    label_to_name = dict(zip(cat_df['Category_Label'], cat_df['Category_Text']))
    user_ids = user_df['Customer_ID'].tolist()

# Sidebar - Instructions
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/263/263115.png", width=100)
st.sidebar.header("🎯 Instructions")
st.sidebar.markdown("""
- Choose a **Customer ID**
- Choose **Top N Recommendations**
- Click **Get Recommendations**
""")

# --- Main Area ---
st.subheader("👤 Select a Customer")
selected_id = st.selectbox("Customer ID", user_ids)

if selected_id:
    user_row = user_df[user_df['Customer_ID'] == selected_id]
    user_input = user_row[user_features].values[0].tolist()

    # Summary Table
    with st.expander("📋 View User Behavior Summary", expanded=False):
        stats = user_row[user_features].T.reset_index()
        stats.columns = ["Feature", "Value"]
        st.dataframe(stats, use_container_width=True)

    # Select Top-K
    top_k = st.selectbox("🔢 How many recommendations?", [1, 3, 5], index=2)

    if st.button("🎁 Get Recommended Categories"):
        with st.spinner("✨ Generating recommendations..."):
            preds = predict_top_k(user_input, model, cat_combined, label_to_name, k=top_k)

        st.success(f"🎉 **Top Pick:** {preds[0]}")
        if top_k > 1:
            st.markdown("#### 🏷️ Other Suggestions:")
            for i, p in enumerate(preds[1:], 2):
                st.markdown(f"- {i}. {p}")
