import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'DeepLearning'))

from dl_model import NCFModel
import torch

def load_model(model_path, n_users, n_recipes):
    model = NCFModel(n_users, n_recipes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def get_top_k_recommendations(model, user_id, recipe_ids, k=5):
    import torch

    model.eval()
    user_tensor = torch.tensor([user_id] * len(recipe_ids))
    recipe_tensor = torch.tensor(recipe_ids)

    with torch.no_grad():
        predictions = model(user_tensor, recipe_tensor)
        top_k_idx = torch.topk(predictions, k).indices
        top_k_recipe_ids = recipe_tensor[top_k_idx].numpy()
        top_k_scores = predictions[top_k_idx].numpy()

    return list(zip(top_k_recipe_ids, top_k_scores))