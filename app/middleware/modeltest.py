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
    
    # Format the response to match the desired structure
    formatted_playlists = []

    for playlist in playlist_data:
        # Split the playlist into the header (title), movies, and description
        lines = playlist.split("\n")
        
        playlist_title = lines[0].strip()  # First line is the playlist title
        playlist_description = lines[-1].strip()  # Last line is the playlist description

        # Extract movie titles and descriptions (assuming they are in lines[1:-1])
        movies = []
        for i in range(1, len(lines) - 1):
            if lines[i].startswith("-"):  # Only process lines that are movies
                movie_data = lines[i].strip().split("\n", 1)
                movie_title = movie_data[0][3:].strip()  # Remove the "- X." part
                movie_description = movie_data[1].strip() if len(movie_data) > 1 else ""
                movies.append({"title": movie_title, "description": movie_description})

        # Add the formatted playlist to the result list
        formatted_playlists.append({
            "playlist_title": playlist_title,
            "movies": movies,
            "playlist_description": playlist_description
        })
    
    return formatted_playlists

def generate_playlists(user_prompt):
    # Generate the AI prompt for movie recommendations based on the user's preferences
    prompt = f"""
    User has given the following preferences: "{user_prompt}"
    Based on these preferences, create at least two movie playlists, each containing at least five movies.
    Only include movies from Disney+'s library.
    Do not include any movies that the user has included in their preferences.
    Each playlist should follow this structure:

    (Playlist Name) // This should be the title of the playlist

    - (1). (movie title) 
      (description of movie here)
    - (2). (movie title) 
      (description of movie here)
    - (3). (movie title) 
      (description of movie here)
    - (4). (movie title) 
      (description of movie here)
    - (5). (movie title) 
      (description of movie here)

    (Playlist description here) // A brief description of the playlist

    Please format the playlist as shown above and separate each playlist by a blank line.
    """
    
    # Use OpenAI GPT to generate movie titles and descriptions with playlist names
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the updated model
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=1000,  # Increase token count to fit multiple movies and descriptions
        temperature=0.7
    )
    
    # Print the full response to the console for debugging
    print("Response from OpenAI:", response)

    # Extract the movie titles and descriptions with the playlist names
    playlist_data = response['choices'][0]['message']['content'].strip()
    
    # Split the response into playlists and movie data (Assuming playlists are separated by a blank line)
    playlists = playlist_data.split('\n\n')  # Assuming playlists are separated by a blank line
    
    return playlists

