from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return RedirectResponse(url='/static/index.html')
