#preprocess_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer

csv_path = "data\Online_Shopping_Data.csv"
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], format='%d/%m/%Y')

    # Clean numeric columns
    df['Total_Spent'] = pd.to_numeric(df['Total_Spent'], errors='coerce')
    df['Price_Per_Unit'] = pd.to_numeric(df['Price_Per_Unit'].astype(str).str.replace(',', '.'), errors='coerce')
    df.dropna(subset=['Customer_ID', 'Category'], inplace=True)

    # Encode categories
    le_category = LabelEncoder()
    df['Category_Label'] = le_category.fit_transform(df['Category'])

    # Feature Engineering
    latest_date = df['Transaction_Date'].max()
    user_df = df.groupby('Customer_ID').agg({
        'Total_Spent': ['mean', 'sum', 'std'],
        'Quantity': ['mean', 'sum'],
        'Price_Per_Unit': ['mean', 'std'],
        'Transaction_Date': lambda x: (latest_date - x.max()).days,
        'Category': pd.Series.nunique,
        'Item': pd.Series.nunique,
        'Location': pd.Series.nunique,
        'Payment_Method': pd.Series.nunique,
        'Category_Label': lambda x: x.iloc[-1]
    }).reset_index()

    user_df.columns = [
        'Customer_ID',
        'Spent_mean', 'Spent_sum', 'Spent_std',
        'Qty_mean', 'Qty_sum',
        'Price_mean', 'Price_std',
        'Recency',
        'Category_Diversity', 'Item_Diversity',
        'Location_Diversity', 'Payment_Diversity',
        'Target_Category'
    ]

    user_features = [
        'Spent_mean', 'Spent_sum', 'Spent_std',
        'Qty_mean', 'Qty_sum',
        'Price_mean', 'Price_std',
        'Recency',
        'Category_Diversity', 'Item_Diversity',
        'Location_Diversity', 'Payment_Diversity'
    ]

    scaler_user = StandardScaler()
    user_df[user_features] = scaler_user.fit_transform(user_df[user_features])

    # Category features
    cat_df = df.groupby('Category_Label').agg({
        'Total_Spent': ['mean'],
        'Quantity': ['mean'],
        'Price_Per_Unit': ['mean'],
        'Customer_ID': 'count'
    }).reset_index()
    cat_df.columns = ['Category_Label', 'Spent_mean', 'Qty_mean', 'Price_mean', 'Popularity']
    cat_df['Category_Text'] = le_category.inverse_transform(cat_df['Category_Label'])

    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    cat_text_vecs = sbert.encode(cat_df['Category_Text'].tolist())

    cat_features = ['Spent_mean', 'Qty_mean', 'Price_mean', 'Popularity']
    scaler_cat = StandardScaler()
    cat_df[cat_features] = scaler_cat.fit_transform(cat_df[cat_features])
    cat_numeric = cat_df[cat_features].values
    cat_combined = np.hstack([cat_numeric, cat_text_vecs])

    cat_lookup_array = {c: cat_combined[i] for i, c in enumerate(cat_df['Category_Label'])}

    return user_df, cat_df, cat_combined, cat_lookup_array, user_features, le_category
