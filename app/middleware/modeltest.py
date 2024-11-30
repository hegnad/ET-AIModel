import os
import openai
from dotenv import load_dotenv
import requests

load_dotenv()  # Load environment variables from .env file

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API client
openai.api_key = OPENAI_API_KEY

def model_test(user_prompt):
    # Step 1: Generate movie titles and descriptions based on user prompt
    playlist_data = generate_playlists(user_prompt)
    
    return playlist_data

def generate_playlists(user_prompt):
    # Generate the AI prompt for movie recommendations based on the user's preferences
    prompt = f"""
    User has given the following preferences: "{user_prompt}"
    Based on these preferences, create at least two movie playlists, each containing at least five movies.
    Each playlist should have a title and a brief explanation of why the movies are recommended.
    Also, only include movies that are available on Disney+.
    Only return the movie titles and descriptions with the playlist names, one per line.
    """
    
    # Use OpenAI GPT to generate movie titles and descriptions with playlist names
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the updated model
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=1000,  # Increase token count to fit multiple movies and descriptions
        temperature=0.7
    )
    
    # Extract the movie titles and descriptions with the playlist names
    playlist_data = response['choices'][0]['message']['content'].strip()
    
    # Split the response into playlists and movie data (Assuming each playlist is formatted as a separate block)
    playlists = playlist_data.split('\n\n')  # Assuming playlists are separated by a blank line
    
    return playlists
