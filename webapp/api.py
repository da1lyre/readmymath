import io
import os
import uuid
import asyncio
import httpx
import cv2
import numpy as np
import base64
from pathlib import Path
from scipy.spatial import cKDTree
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from webapp.db import init_db, save_attempt, get_history, get_average_score

app = FastAPI(title="WriteYourMath API")

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
TEMPLATES_DIR = WEB_DIR / "templates"


@app.on_event("startup")
async def startup():
    await asyncio.sleep(3)
    await init_db()


def preprocess_for_ocr(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    img = img.convert("L")
    bbox = img.point(lambda p: 255 if p < 200 else 0).getbbox()

    if bbox:
        img = img.crop(bbox)
        padding = 60
        new_img = Image.new("L", (img.width + padding * 2, img.height + padding * 2), 255)
        new_img.paste(img, (padding, padding))
        img = new_img

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/")
async def root():
    return {"status": "ok", "crispembed": CRISPEMBED_URL}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="PNG or JPEG required")

    png = await file.read()
    standardized_png = preprocess_for_ocr(png)

    debug_path = SHARED_DIR / "debug_ocr.png"
    debug_path.write_bytes(standardized_png)

    filename = f"{uuid.uuid4().hex}.png"
    shared_path = SHARED_DIR / filename
    shared_path.write_bytes(standardized_png)

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

    return {"latex": latex}


@app.post("/compare")
async def compare(request: Request, file: UploadFile = File(...), char_code: str = ""):
    png = await file.read()

    template_path = TEMPLATES_DIR / f"u{char_code}.png"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")

    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    _, template = cv2.threshold(template, 200, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(template)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        template = template[y:y + h, x:x + w]
        max_side = max(w, h)
        scale = 200 / max_side
        template = cv2.resize(template, (int(w * scale), int(h * scale)))
        canvas = np.zeros((256, 256), dtype=np.uint8)
        px = (256 - template.shape[1]) // 2
        py = (256 - template.shape[0]) // 2
        canvas[py:py + template.shape[0], px:px + template.shape[1]] = template
        template = canvas
    else:
        template = np.zeros((256, 256), dtype=np.uint8)

    img_pil = Image.open(io.BytesIO(png))
    if img_pil.mode in ('RGBA', 'LA') or (img_pil.mode == 'P' and 'transparency' in img_pil.info):
        background = Image.new("RGB", img_pil.size, (255, 255, 255))
        background.paste(img_pil, mask=img_pil.split()[-1])
        img_pil = background
    img_pil = img_pil.convert("L")

    user = np.array(img_pil)
    _, user = cv2.threshold(user, 200, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(user)
    if coords is None:
        return {"score": 0, "feedback": "You didn't draw anything!"}

    x, y, w, h = cv2.boundingRect(coords)
    user = user[y:y + h, x:x + w]

    max_side = max(w, h)
    scale = 200 / max_side
    user = cv2.resize(user, (int(w * scale), int(h * scale)))
    user_canvas = np.zeros((256, 256), dtype=np.uint8)
    px = (256 - user.shape[1]) // 2
    py = (256 - user.shape[0]) // 2
    user_canvas[py:py + user.shape[0], px:px + user.shape[1]] = user
    user = user_canvas

    overlay = np.zeros((256, 256, 3), dtype=np.uint8)
    intersect_mask = np.logical_and(template > 0, user > 0)
    overlay[intersect_mask] = [0, 255, 0]
    user_only_mask = np.logical_and(user > 0, template == 0)
    overlay[user_only_mask] = [255, 0, 0]
    template_only_mask = np.logical_and(template > 0, user == 0)
    overlay[template_only_mask] = [0, 0, 255]

    _, buffer = cv2.imencode('.png', overlay)
    overlay_base64 = base64.b64encode(buffer).decode('utf-8')

    intersection = np.logical_and(template > 0, user > 0).sum()
    union = np.logical_or(template > 0, user > 0).sum()
    iou = intersection / union if union > 0 else 0.0

    template_points = np.column_stack(np.where(template > 0))
    user_points = np.column_stack(np.where(user > 0))

    if len(user_points) == 0 or len(template_points) == 0:
        return {"score": 0, "feedback": "Drawing is empty."}

    tree = cKDTree(template_points)
    distances, _ = tree.query(user_points)
    chamfer = float(np.mean(distances))


    chamfer_score = max(0, 1 - (chamfer / 100))

    normalized_iou = min(1.0, iou / 0.7)
    final_score = int(100 * (0.6 * normalized_iou + 0.4 * chamfer_score))

    final_score = min(100, final_score)

    if final_score >= 90:
        feedback = "Perfect! 🎯"
    elif final_score >= 70:
        feedback = "Great! Very close to the template."
    elif final_score >= 50:
        feedback = "Good, but there are some deviations."
    else:
        feedback = "Too far from the template, try again."

    user_id = request.query_params.get("user_id", "test_user")
    await save_attempt(user_id, char_code, float(final_score))

    return {
        "score": final_score,
        "feedback": feedback,
        "overlay": overlay_base64
    }

@app.get("/history")
async def history(request: Request):
    user_id = request.query_params.get("user_id", "test_user")
    attempts = await get_history(user_id, limit=10)
    avg = await get_average_score(user_id)

    return {
        "average_score": avg,
        "attempts": [
            {
                "char": chr(int(a.char_code, 16)) if a.char_code else "?",
                "score": a.score,
                "date": a.created_at.strftime("%H:%M")
            } for a in attempts
        ]
    }