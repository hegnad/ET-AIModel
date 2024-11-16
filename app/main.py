from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.middleware.processPrompt import promptResponse
from app.middleware.modeltest import model_test
from app.middleware.playlist_generator import generate_playlist

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def process_form(
    request: Request, genres: str = Form(...), recent: str = Form(None)
):
    # Generate a playlist based on user input
    playlist = generate_playlist(genres, recent)
    return templates.TemplateResponse(
        "form.html", {"request": request, "response": playlist}
    )