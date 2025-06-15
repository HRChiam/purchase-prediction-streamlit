# src/inference.py
import torch
import numpy as np

def predict_top_k(user_features_input, model, cat_vecs, label_to_name, k=3):
    with torch.no_grad():
        u_tensor = torch.tensor(user_features_input, dtype=torch.float32).unsqueeze(0)
        u_emb = model.user_net(u_tensor)
        c_emb = model.cat_net(torch.tensor(cat_vecs, dtype=torch.float32))
        sims = u_emb @ c_emb.T
        top_k_indices = torch.topk(sims, k=k, dim=1).indices[0].cpu().numpy()
        return [label_to_name[i] for i in top_k_indices]
