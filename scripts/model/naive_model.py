import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def run_naive_model():
    df = pd.read_csv("C:/DeepLearning/FinalProject/scripts/data/processed/cleaned_recipes.csv")
    
    
    mean_rating = df["rating"].mean()
    df["naive_prediction"] = mean_rating
    
    # Eval
    rmse = np.sqrt(mean_squared_error(df["rating"], df["naive_prediction"]))
    mae = mean_absolute_error(df["rating"], df["naive_prediction"])

    print(f"Naive Model — RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae

if __name__ == "__main__":
    run_naive_model()
