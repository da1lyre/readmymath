from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="ReadMyMath API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=WEB_DIR, html=True), name="static")


@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    # Заглушка. Позже — вызов BTTR/HMER.
    return {"latex": "x+1", "raw": "stub"}