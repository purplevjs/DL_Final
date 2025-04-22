# DL_Final
# 🥦 Smart Nutrition Recipe Recommender

This project is a personalized recipe recommendation system that uses collaborative filtering and deep learning to suggest healthy meals based on a user's preferences, health goals, and dietary restrictions.

---

## 📌 Features

- 🍽 Recommends recipes tailored to your health goals and allergies
- 🤖 Supports 3 modeling approaches:
  - Naive mean model
  - Classical ML (Linear Regression)
  - Deep Learning (Neural Collaborative Filtering)
- 🧠 Personalized predictions using user-recipe interaction data
- 🌐 Interactive web app built with **Streamlit**
- 📸 Recipe images included via static/dynamic URL mapping

---

## 🔍 Problem Statement

Many people struggle to find meals that align with their health goals, such as reducing sugar or increasing protein. This recommender system helps users discover recipes that match their personal health needs using intelligent algorithms.

---

## 🧬 Dataset

- 📂 Source: [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- Includes:
  - 230K+ recipes
  - 180K+ user interactions (ratings, reviews)
- Cleaned and processed for use in recommendation models

---

## ⚙️ Project Structure
FinalProject/ ├── app.py # Streamlit app ├── requirements.txt # Python dependencies ├── README.md # Project overview ├── .gitignore # Files to ignore ├── data/ │ ├── raw/ # Raw input data (not tracked) │ └── processed/ # Cleaned & structured data ├── scripts/ │ ├── make_dataset.py # Preprocessing │ └── model/ │ ├── naive_model.py # Mean rating model │ ├── classical_model.py # Linear regression model │ └── DeepLearning/ │ ├── dl_model.py # NCF model architecture │ ├── train_model.py # Deep model training │ └── recommend.py # Recommendation logic


## Preprocess the dataset:
python scripts/make_dataset.py


## Train the deep learning model:
python scripts/model/DeepLearning/train_model.py


## Launch the Streamlit app:
streamlit run app.py


## 📈 Evaluation Metrics

Model	Metric	Value
Naive	RMSE	~1.30
Classical (LR)	RMSE	~1.12
Deep Learning	RMSE	~0.97
MAE also tracked to measure performance.

DL model provides better personalization.

## 💡 Results & Insights
✅ Deep learning achieved the best prediction accuracy.

✅ Personalized filtering improves dietary goal alignment.

✅ Recipes are matched with user taste and health profile.

✅ Future improvement: Real-time feedback loop integration.

## ⚖️ Ethics Statement
No sensitive user data used.

Dataset is public (Kaggle).

Recommendation fairness and health claims will be refined in future work.


