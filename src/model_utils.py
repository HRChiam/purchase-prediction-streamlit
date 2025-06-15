#model_utils.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ContrastiveModel(nn.Module):
    def __init__(self, input_user, input_cat, emb_dim=64):
        super().__init__()
        self.user_net = nn.Sequential(
            nn.Linear(input_user, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )
        self.cat_net = nn.Sequential(
            nn.Linear(input_cat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, emb_dim)
        )

    def forward(self, u, c):
        u_emb = self.user_net(u)
        c_emb = self.cat_net(c)
        return u_emb, c_emb

class ContrastiveDataset(Dataset):
    def __init__(self, user_feats, cat_vecs, user_to_cat, user_ids, cat_names):
        self.user_feats = user_feats
        self.cat_vecs = cat_vecs
        self.user_to_cat = user_to_cat
        self.user_ids = user_ids
        self.cat_names = cat_names

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        u_feat = self.user_feats[idx]
        pos_cat = self.user_to_cat.get(user_id, np.random.choice(self.cat_names))
        pos_cat_idx = np.where(self.cat_names == pos_cat)[0][0]
        c_feat = self.cat_vecs[pos_cat_idx]
        return torch.tensor(u_feat, dtype=torch.float32), torch.tensor(c_feat, dtype=torch.float32)
