import json
import torch
import os
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.middleware.bert4rec import BERT4Rec, generate_recommendations

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

# Load pre-trained model and movie encoder
def load_resources():
    """
    Load model and encoder resources.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    # Load movie encoder
    with open(os.path.join(data_dir, "movie_encoder.json"), "r") as encoder_file:
        movie_classes = json.load(encoder_file)

    encoder = {idx: title for idx, title in enumerate(movie_classes)}
    decoder = {title: idx for idx, title in enumerate(movie_classes)}

    # Load trained model
    model = BERT4Rec(num_items=len(movie_classes))
    model.load_state_dict(torch.load(os.path.join(data_dir, "bert4rec_model.pth")))
    model.eval()

    return model, encoder, decoder

def load_users():
    """
    Load user data from JSON file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    with open(os.path.join(data_dir, "users_mock_catalog.json"), "r") as user_file:
        return json.load(user_file)

# Initialize resources
bert_model, movie_encoder, movie_decoder = load_resources()
users = load_users()

def load_movies():
    """
    Load movie data from JSON file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    with open(os.path.join(data_dir, "mock_catalog.json"), "r") as movie_file:
        return json.load(movie_file)

movies = load_movies()

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """
    Display the form for generating recommendations.
    """
    return templates.TemplateResponse("form.html", {"request": request})

@app.get("/user-data/{user_id}")
async def get_user_data(user_id: int):
    """
    Retrieve the watchlist and watched movies for a specific user.

    Args:
        user_id (int): The user's ID.

    Returns:
        dict: A JSON response with the user's watchlist and watched history.
    """
    # Get user data
    user = next((user for user in users if user["user_id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "watchlist": user["watchlist"],
        "watched": user["watched"]
    }

@app.post("/recommend")
async def recommend(user_id: int = Form(...)):
    """
    Recommend movies to a user based on their watched history.

    Args:
        user_id (int): The user's ID.

    Returns:
        dict: A JSON response with recommended movie titles, popularity scores, and explanations.
    """
    # Get user data
    user = next((user for user in users if user["user_id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Ensure the user has a watched history
    if not user["watched"]:
        raise HTTPException(status_code=400, detail="User has no watched history")

    # Encode user's watched sequence
    user_sequence = [movie_decoder[movie] for movie in user["watched"] if movie in movie_decoder]

    # Check if the user has watched enough movies for recommendations
    if not user_sequence:
        raise HTTPException(status_code=400, detail="Not enough watched data to generate recommendations")

    # Generate recommendations
    recommended_ids = generate_recommendations(bert_model, user_sequence)

    # Filter out movies the user has already watched
    watched_ids = set(user_sequence)
    filtered_recommendations = [
        movie_id for movie_id in recommended_ids if movie_id not in watched_ids
    ]

    # Map recommendations back to titles and add metadata
    recommendations = []
    for movie_id in filtered_recommendations[:10]:  # Limit to top 10
        movie_title = movie_encoder[movie_id]
        movie_details = next((movie for movie in movies if movie["title"] == movie_title), None)
        if movie_details:
            explanation = []
            if movie_title in user["watched"]:
                explanation.append("Based on your watch history")
            if movie_title in user["likes"]:
                explanation.append("You liked similar movies")
            if movie_title in user["watchlist"]:
                explanation.append("On your watchlist")
            if movie_title in user["ratings"]:
                explanation.append(f"High rating ({user['ratings'][movie_title]}/5)")

            recommendations.append({
                "title": movie_title,
                "popularity": movie_details["popularity"],
                "explanation": ", ".join(explanation) if explanation else "Recommended by model",
            })

    return {"recommendations": recommendations}

