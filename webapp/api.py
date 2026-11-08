from pathlib import Path
import io
import os
import uuid
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI(title="ReadMyMath API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=WEB_DIR, html=True), name="static")

CRISPEMBED_URL = os.getenv("CRISPEMBED_SERVER_URL", "http://crispembed:8080")
SHARED_DIR = Path("/shared")
SHARED_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    return {"status": "ok", "crispembed": CRISPEMBED_URL}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Нужен PNG или JPEG")

    png = await file.read()
    if len(png) < 100:
        raise HTTPException(status_code=400, detail="Файл слишком маленький")

    # Сохраняем в общую папку
    filename = f"{uuid.uuid4().hex}.png"
    shared_path = SHARED_DIR / filename
    shared_path.write_bytes(png)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{CRISPEMBED_URL}/math/ocr",
                json={"image": str(shared_path)},
            )
            resp.raise_for_status()
            data = resp.json()

        latex = data.get("text") or data.get("latex") or data.get("result") or ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"CrispEmbed error: {e}")
    finally:
        try:
            shared_path.unlink()
        except OSError:
            pass

    return {"latex": latex, "model": "posformer"}