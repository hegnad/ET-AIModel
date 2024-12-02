# FastAPI + OpenAI Movie Playlist Curator

This project leverages FastAPI to create a web application that allows users to input their preferences and generate personalized movie playlists using OpenAI's GPT model. The playlists are curated based on the user’s input and returned as suggestions in a clean, readable format.

Features:

    Personalized Movie Playlists: Users can input their preferences, and the app generates a list of movie recommendations, with descriptions and themed playlists.
    FastAPI Backend: A lightweight and fast backend that communicates with OpenAI’s GPT-3.5 model to generate the movie playlists.
    Dynamic Frontend: A simple frontend that displays the generated playlists in a clear and user-friendly way.

Technologies:

    Backend: FastAPI, OpenAI API
    Frontend: HTML, Jinja2, CSS
    Environment Management: Python dotenv for managing environment variables
    CSS: Custom styles for displaying playlists

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/hegnad/ET-AIModel/tree/newmodel.git
cd ET-AIModel

```

### 2. Install all dependencies using either of the following:

pip install -r requirements.txt

or

pip3 install -r requirements.txt

### 3. Configure OpenAI API:

Create a .env file in the root directory of the project and add your OpenAI API key:

```makefile
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the application

For Mac:
```bash
uvicorn app.main:app --reload
```


For Windows:
```bash
python.exe -m uvicorn.main app.main:app --reload
```

### 5. Use application in browser

visit http://localhost:8000

## Future Enhancements:

    User History: Implement the ability to personalize playlists based on a user's viewing history.
    Search Functionality: Allow users to search for specific movies or genres to refine their playlists.
    AI Improvement: Explore fine-tuning the OpenAI model for more accurate and diverse playlist recommendations.
