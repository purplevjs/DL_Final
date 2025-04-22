import pandas as pd
import ast
import os

def parse_nutrition(nutrition_str):
    try:
        return ast.literal_eval(nutrition_str)
    except:
        return [0.0] * 7

def load_and_preprocess_data():
    interactions_df = pd.read_csv("C:/DeepLearning/FinalProject/scripts/data/raw/RAW_interactions.csv")
    recipes_df = pd.read_csv("C:/DeepLearning/FinalProject/scripts/data/raw/RAW_recipes.csv")

    interactions_df = interactions_df[["user_id", "recipe_id", "rating"]].dropna()
    recipes_df = recipes_df[["id", "name", "minutes", "tags", "nutrition", "steps"]].dropna()
    recipes_df = recipes_df.rename(columns={"id": "recipe_id", "name": "recipe_name"})

    recipes_df["nutrition_parsed"] = recipes_df["nutrition"].apply(parse_nutrition)
    recipes_df[["calories", "fat", "sugar", "protein"]] = pd.DataFrame(
        recipes_df["nutrition_parsed"].to_list(), index=recipes_df.index
    )[[0, 1, 2, 3]]
    recipes_df = recipes_df.drop(columns=["nutrition", "nutrition_parsed"])

    merged_df = pd.merge(interactions_df, recipes_df, on="recipe_id", how="inner")
    return merged_df

if __name__ == "__main__":
    df = load_and_preprocess_data()
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("C:/DeepLearning/FinalProject/scripts/data/processed/cleaned_recipes.csv", index=False)
    print("Cleaned dataset saved.")
