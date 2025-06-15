# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess_utils import load_and_preprocess_data
from model_utils import ContrastiveModel, ContrastiveDataset

csv_path = "data/Online_Shopping_Data.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

user_df, cat_df, cat_combined, cat_lookup_array, user_features, le_category = load_and_preprocess_data(csv_path)

user_to_cat = dict(zip(user_df['Customer_ID'], user_df['Target_Category']))
user_feats = user_df[user_features].values
user_ids = user_df['Customer_ID'].values
cat_names = cat_df['Category_Label'].values
cat_vecs = cat_combined

dataset = ContrastiveDataset(user_feats, cat_vecs, user_to_cat, user_ids, cat_names)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ContrastiveModel(input_user=len(user_features), input_cat=cat_combined.shape[1], emb_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CosineEmbeddingLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for u_feat, c_feat in dataloader:
        u_feat, c_feat = u_feat.to(device), c_feat.to(device)
        optimizer.zero_grad()
        u_emb, c_emb = model(u_feat, c_feat)
        target = torch.ones(u_emb.size(0)).to(device)
        loss = criterion(u_emb, c_emb, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
