import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
from dl_model import NCFModel

df = pd.read_csv("C:/DeepLearning/FinalProject/scripts/data/processed/cleaned_recipes.csv")
df["user"] = LabelEncoder().fit_transform(df["user_id"])
df["recipe"] = LabelEncoder().fit_transform(df["recipe_id"])

class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user"].values, dtype=torch.long)
        self.recipes = torch.tensor(df["recipe"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self): return len(self.ratings)
    def __getitem__(self, idx): return self.users[idx], self.recipes[idx], self.ratings[idx]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_loader = DataLoader(RatingsDataset(train_df), batch_size=128, shuffle=True)
test_loader = DataLoader(RatingsDataset(test_df), batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NCFModel(df["user"].nunique(), df["recipe"].nunique()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    for users, recipes, ratings in train_loader:
        users, recipes, ratings = users.to(device), recipes.to(device), ratings.to(device)
        optimizer.zero_grad()
        preds = model(users, recipes)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")


os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "C:/DeepLearning/FinalProject/scripts/model/model_ncf.pth")

# Evaluate
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for users, recipes, ratings in test_loader:
        users, recipes = users.to(device), recipes.to(device)
        preds = model(users, recipes)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(ratings.numpy())

rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
mae = mean_absolute_error(all_targets, all_preds)
print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")
