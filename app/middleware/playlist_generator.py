import json

# Load mock data
with open("data/mock_catalog.json", "r") as file:
    content_catalog = json.load(file)

def generate_playlist(preferred_genres: str, recently_watched: str = None) -> list:
    """
    Generate a curated playlist based on preferred genres and recently watched titles.
    
    Args:
        preferred_genres (str): Comma-separated string of user’s favorite genres.
        recently_watched (str): Comma-separated string of recently watched titles.
    
    Returns:
        list: Curated playlist of recommended content.
    """
    genres = [genre.strip().lower() for genre in preferred_genres.split(",")]
    recommendations = []

    # Filter catalog based on preferred genres
    for content in content_catalog:
        content_genres = [g.lower() for g in content["genre"]]
        if any(genre in content_genres for genre in genres):
            recommendations.append(content)
    
    # Sort recommendations by popularity
    recommendations.sort(key=lambda x: x["popularity"], reverse=True)

    # Format recommendations for display
    return [f"{item['title']} (Genre: {', '.join(item['genre'])})" for item in recommendations]
