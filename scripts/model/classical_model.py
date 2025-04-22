import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def run_classical_model():
    df = pd.read_csv("C:/DeepLearning/FinalProject/scripts/data/processed/cleaned_recipes.csv")
    
    features = ["calories", "fat", "sugar", "protein", "minutes"]
    X = df[features]
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"Classical Model — RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae

if __name__ == "__main__":
    run_classical_model()
