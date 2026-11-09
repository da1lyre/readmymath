import io
import os
import uuid
import asyncio
import httpx
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from webapp.db import (
    init_db, save_attempt, get_history, get_average_score,
    get_char_stats, get_total_attempts, get_high_score, update_high_score, save_formula, get_saved_formulas
)

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

EXPRESSION_LATEX = {
    "exp_1": "x+1", "exp_2": "2a", "exp_3": "y=5", "exp_4": "a-b",
    "exp_9": "3x", "exp_10": "b=2", "exp_11": "x+y", "exp_12": "m-n",
    "exp_13": "5-2", "exp_14": "4c", "exp_15": "p=q", "exp_16": "7+8",
    "exp_5": "\\frac{a}{b}", "exp_6": "x^2", "exp_7": "\\sqrt{x}", "exp_8": "2x+3",
    "exp_17": "y^3", "exp_18": "\\frac{1}{x}", "exp_19": "\\sqrt{y}", "exp_20": "a^2+b^2",
    "exp_21": "\\frac{x}{2}", "exp_22": "3x-1", "exp_23": "\\sqrt{x+1}", "exp_24": "x^2=4",
    "imp_1": "\\int_0^1 x^2 \\, dx", "imp_2": "\\lim_{x \\to 0} \\frac{\\sin x}{x}",
    "imp_3": "\\sum_{i=1}^n i^2", "imp_4": "e^{i\\pi} + 1 = 0"
}


@app.on_event("startup")
async def startup():
    await asyncio.sleep(3)
    await init_db()


def check_empty_canvas(image_bytes: bytes) -> bool:
    if not image_bytes or len(image_bytes) < 100:
        return True
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        img = img.convert("L")
        return img.point(lambda p: 255 if p < 200 else 0).getbbox() is None
    except Exception:
        return True


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
        w, h = img.size
        target_ratio = 3 / 2

        if w / h < target_ratio:
            new_w = int(h * target_ratio)
            new_img = Image.new("L", (new_w, h), 255)
            paste_x = (new_w - w) // 2
            new_img.paste(img, (paste_x, 0))
            img = new_img
        else:
            new_h = int(w / target_ratio)
            new_img = Image.new("L", (w, new_h), 255)
            paste_y = (new_h - h) // 2
            new_img.paste(img, (0, paste_y))
            img = new_img

        padding = 60
        final_img = Image.new("L", (img.width + padding * 2, img.height + padding * 2), 255)
        final_img.paste(img, (padding, padding))
        img = final_img

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compare_images(template_path: Path, user_bytes: bytes, resize_to_square=True):
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        return 0, "Template not found"

    _, template = cv2.threshold(template, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(template)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        template = template[y:y + h, x:x + w]
        if resize_to_square:
            template = cv2.resize(template, (256, 256))
        else:
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

    img_pil = Image.open(io.BytesIO(user_bytes))
    if img_pil.mode in ('RGBA', 'LA') or (img_pil.mode == 'P' and 'transparency' in img_pil.info):
        background = Image.new("RGB", img_pil.size, (255, 255, 255))
        background.paste(img_pil, mask=img_pil.split()[-1])
        img_pil = background
    img_pil = img_pil.convert("L")

    user = np.array(img_pil)
    _, user = cv2.threshold(user, 200, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(user)
    if coords is None:
        return 0, "Drawing is empty."

    x, y, w, h = cv2.boundingRect(coords)
    user = user[y:y + h, x:x + w]

    if resize_to_square:
        user = cv2.resize(user, (256, 256))
    else:
        max_side = max(w, h)
        scale = 200 / max_side
        user = cv2.resize(user, (int(w * scale), int(h * scale)))
        user_canvas = np.zeros((256, 256), dtype=np.uint8)
        px = (256 - user.shape[1]) // 2
        py = (256 - user.shape[0]) // 2
        user_canvas[py:py + user.shape[0], px:px + user.shape[1]] = user
        user = user_canvas

    intersection = np.logical_and(template > 0, user > 0).sum()
    union = np.logical_or(template > 0, user > 0).sum()
    iou = intersection / union if union > 0 else 0.0

    template_points = np.column_stack(np.where(template > 0))
    user_points = np.column_stack(np.where(user > 0))

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

    return final_score, feedback


@app.get("/")
async def root():
    return {"status": "ok", "crispembed": CRISPEMBED_URL}


@app.post("/recognize")
async def recognize(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="PNG or JPEG required")
    png = await file.read()

    if check_empty_canvas(png):
        raise HTTPException(status_code=400, detail="Canvas is empty")

    standardized_png = preprocess_for_ocr(png)
    filename = f"{uuid.uuid4().hex}.png"
    shared_path = SHARED_DIR / filename
    shared_path.write_bytes(standardized_png)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{CRISPEMBED_URL}/math/ocr", json={"image": str(shared_path)})
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
    user_id = request.query_params.get("user_id", "test_user")
    attempt_id = await save_attempt(user_id, "formula", latex, 100.0)

    return {"latex": latex, "attempt_id": attempt_id}


@app.post("/compare")
async def compare(request: Request, file: UploadFile = File(...), char_code: str = ""):
    png = await file.read()
    if check_empty_canvas(png):
        raise HTTPException(status_code=400, detail="Canvas is empty")

    template_path = TEMPLATES_DIR / f"u{char_code}.png"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")

    final_score, feedback = compare_images(template_path, png, resize_to_square=False)

    user_id = request.query_params.get("user_id", "test_user")
    char_text = chr(int(char_code, 16)) if char_code else "?"
    await save_attempt(user_id, char_code, char_text, float(final_score))

    return {"score": final_score, "feedback": feedback}


@app.post("/compare_expression")
async def compare_expression(request: Request, file: UploadFile = File(...), exp_id: str = ""):
    png = await file.read()
    if check_empty_canvas(png):
        raise HTTPException(status_code=400, detail="Canvas is empty")

    template_path = TEMPLATES_DIR / f"{exp_id}.png"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Expression template not found")

    final_score, feedback = compare_images(template_path, png, resize_to_square=True)

    user_id = request.query_params.get("user_id", "test_user")
    latex_str = EXPRESSION_LATEX.get(exp_id, "")
    await save_attempt(user_id, exp_id, latex_str, float(final_score))

    return {"score": final_score, "feedback": feedback}


@app.get("/history")
async def history(request: Request):
    user_id = request.query_params.get("user_id", "test_user")
    limit = int(request.query_params.get("limit", 10))

    attempts = await get_history(user_id, limit=limit)
    avg = await get_average_score(user_id)
    total = await get_total_attempts(user_id)

    return {
        "average_score": avg,
        "total_attempts": total,
        "attempts": [
            {
                "char": a.latex_str if a.latex_str else a.char_code,
                "score": a.score,
                "date": a.created_at.strftime("%H:%M")
            } for a in attempts
        ]
    }


@app.get("/char_stats")
async def char_stats(request: Request, char_code: str = ""):
    user_id = request.query_params.get("user_id", "test_user")
    return await get_char_stats(user_id, char_code)


@app.get("/highscore")
async def fetch_highscore(request: Request):
    user_id = request.query_params.get("user_id", "test_user")
    return {"highscore": await get_high_score(user_id)}


@app.post("/highscore")
async def save_highscore(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "test_user")
    new_score = data.get("score", 0)
    return {"highscore": await update_high_score(user_id, new_score)}


@app.post("/save_formula")
async def api_save_formula(request: Request):
    data = await request.json()
    formula_id = data.get("attempt_id")
    if formula_id:
        await save_formula(int(formula_id))
    return {"status": "ok"}

@app.get("/saved_formulas")
async def api_saved_formulas(request: Request):
    user_id = request.query_params.get("user_id", "test_user")
    formulas = await get_saved_formulas(user_id)
    return {
        "formulas": [
            {
                "id": f.id,
                "latex": f.latex_str,
                "date": f.created_at.strftime("%Y-%m-%d %H:%M")
            } for f in formulas
        ]
    }


@app.post("/send_to_chat")
async def send_to_chat(
        user_id: str = Form(...),
        photo: UploadFile = File(...)
):
    BOT_URL = "http://bot:8081/send_photo"

    try:
        photo_bytes = await photo.read()

        async with httpx.AsyncClient() as client:
            files = {'photo': ('formula.png', photo_bytes, 'image/png')}
            data = {'user_id': user_id}

            resp = await client.post(BOT_URL, data=data, files=files, timeout=30.0)
            resp.raise_for_status()

        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bot error: {e}")