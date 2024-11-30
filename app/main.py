from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.middleware.modeltest import model_test

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def process_form(request: Request, user_prompt: str = Form(...)):
    # Call the model to generate the playlists based on the user prompt
    playlists = model_test(user_prompt)
    return templates.TemplateResponse("form.html", {"request": request, "response": playlists})
