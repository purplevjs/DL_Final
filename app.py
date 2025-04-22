import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder
import torch


st.set_page_config(page_title="Healthy Recipe Recommender", layout="wide")

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'model', 'DeepLearning'))
from dl_model import NCFModel
from recommend import load_model


st.markdown("""
<style>
  html, body { background-color: #101010; color: #fff; }
  .block-container { padding: 2rem 5rem; font-family: 'Segoe UI', sans-serif; }
  h1 { color: #70e000; font-size: 2.5rem; font-weight: 800; }
  .recipe-card { background: #1b4332; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.4); transition: transform .3s; }
  .recipe-card:hover { transform: translateY(-4px); }
  .recipe-title { font-size: 1.4rem; color: #d8f3dc; margin-bottom: .5rem; }
  .subinfo { color: #e9ecef; font-size: .9rem; }
  .stButton>button { background: linear-gradient(90deg,#70e000,#38b000); border-radius: 8px; font-size: 1rem; font-weight: 600; padding: .5rem 1.5rem; }
</style>
""", unsafe_allow_html=True)


df = pd.read_csv("data/processed/cleaned_recipes.csv")


dedup_df = df.drop_duplicates(subset=["recipe_id"]).reset_index(drop=True)

# Encoding
recipe_encoder = LabelEncoder()
dedup_df["recipe_idx"] = recipe_encoder.fit_transform(dedup_df["recipe_id"])
user_encoder = LabelEncoder()
df["user_idx"] = user_encoder.fit_transform(df["user_id"])

# image mapping
image_urls = {
    "quinoa salad with black beans and mango": "https://source.unsplash.com/400x300/?quinoa",
    "fire roasted salsa ms salsa por favor": "https://source.unsplash.com/400x300/?salsa",
    "applesauce": "https://source.unsplash.com/400x300/?applesauce",
    "0 carb 0 cal gummy worms": "https://source.unsplash.com/400x300/?gummy",
    "0 point ice cream only 1 ingredient": "https://source.unsplash.com/400x300/?ice-cream",
    "0 fat chunky watermelon salsa": "https://source.unsplash.com/400x300/?watermelon"
}

def generate_image_url(recipe_name):
    return image_urls.get(recipe_name.lower(), f"https://source.unsplash.com/400x300/?{recipe_name.replace(' ', '-')}")

# load model
@st.cache_resource
def load_ncf_model():
    num_users = df['user_idx'].nunique()
    num_items = dedup_df['recipe_idx'].nunique()
    return load_model("scripts/model/model_ncf.pth", num_users, num_items)
model = load_ncf_model()


def show_card(title, cal, prot, sugar, fat, score):
    cols = st.columns([1, 3])
    with cols[0]:
        st.image(generate_image_url(title), width=220)
    with cols[1]:
        st.markdown(f"### {title}")
        st.markdown(f"⭐ **Rating:** {score:.2f}")
        st.markdown(f"🔥 **Calories:** {cal} | 🧪 **Protein:** {prot}g | 🍬 **Sugar:** {sugar}g | 🥩 **Fat:** {fat}g")
        st.markdown("---")


st.markdown("# 🥦 Smart Nutrition Recipe Recommender")

# User Input
with st.form("user_preferences_form"):
    goal = st.selectbox("🍀 Health Goal", ["Weight Loss", "Muscle Gain", "Low Sugar", "Balanced Diet"])
    allergy = st.multiselect("🚫 Allergies", ["Dairy", "Nuts", "Gluten"])
    additional = st.multiselect("📌 Additional Preferences", ["High Protein", "Low Sugar", "Low Fat"])
    user = st.selectbox("🙋‍♀️ Select a User ID", df['user_id'].unique())
    submitted = st.form_submit_button("Recommend Recipes")


if submitted:
    user_idx = user_encoder.transform([user])[0]
    item_ids = dedup_df['recipe_idx'].values
    user_tensor = torch.tensor([user_idx] * len(item_ids))
    item_tensor = torch.tensor(item_ids)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor).cpu().numpy()

    dedup_df['score'] = scores
    top5 = dedup_df.sort_values("score", ascending=False).drop_duplicates(subset='recipe_name').head(5)

    st.markdown("### 🧠 Recommended For You")
    for _, row in top5.iterrows():
        title = row['recipe_name']
        show_card(title, row['calories'], row['protein'], row['sugar'], row['fat'], row['score'])
