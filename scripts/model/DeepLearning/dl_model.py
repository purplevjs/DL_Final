import torch
import torch.nn as nn

class NCFModel(nn.Module):
    def __init__(self, n_users, n_recipes, embed_dim=32):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.recipe_embed = nn.Embedding(n_recipes, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.out = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, user, recipe):
        u = self.user_embed(user)
        r = self.recipe_embed(recipe)
        x = torch.cat([u, r], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x).squeeze()
