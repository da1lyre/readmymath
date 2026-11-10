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
    get_char_stats, get_total_attempts, get_high_score, update_high_score, save_formula, get_saved_formulas,
    get_leaderboard, get_user_rank, get_or_create_user
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
    # Medium
    "exp_1": "x+1", "exp_2": "2a", "exp_3": "y=5", "exp_4": "a-b", "exp_5": "3x", "exp_6": "b=2",
    "exp_7": "x+y", "exp_8": "m-n", "exp_9": "5-2", "exp_10": "4c", "exp_11": "p=q", "exp_12": "7+8",
    "exp_13": "a+b", "exp_14": "x=3", "exp_15": "2y", "exp_16": "4+5", "exp_17": "n-1", "exp_18": "3+a",
    "exp_19": "y=7", "exp_20": "c-d", "exp_21": "8-4", "exp_22": "x-2", "exp_23": "5y", "exp_24": "b+3",
    "exp_25": "z=1", "exp_26": "6+2", "exp_27": "a=4", "exp_28": "m+2", "exp_29": "9-3", "exp_30": "2x+1",
    # Hard
    "exp_31": "\\frac{a}{b}", "exp_32": "x^2", "exp_33": "\\sqrt{x}", "exp_34": "2x+3", "exp_35": "y^3",
    "exp_36": "\\frac{1}{x}", "exp_37": "\\sqrt{y}", "exp_38": "a^2+b^2", "exp_39": "\\frac{x}{2}", "exp_40": "3x-1",
    "exp_41": "\\sqrt{x+1}", "exp_42": "x^2=4", "exp_43": "\\frac{2}{3}", "exp_44": "z^2", "exp_45": "\\sqrt{z}",
    "exp_46": "4x-2", "exp_47": "\\frac{a+c}{b}", "exp_48": "x^3", "exp_49": "\\sqrt{x-1}", "exp_50": "x^2+2x",
    "exp_51": "\\frac{3}{x}", "exp_52": "2x^2", "exp_53": "y^2-1", "exp_54": "\\sqrt{2x}", "exp_55": "5x+2",
    "exp_56": "\\frac{x+1}{2}", "exp_57": "a^3", "exp_58": "\\sqrt{x^2}", "exp_59": "4x^2", "exp_60": "\\frac{1}{x^2}",
    "exp_61": "x=2^3", "exp_62": "\\sqrt{4}", "exp_63": "7y-3", "exp_64": "\\frac{y}{4}", "exp_65": "3x^2-5",
    "exp_66": "\\sqrt{x+2}", "exp_67": "x^2-9", "exp_68": "\\frac{x}{y}", "exp_69": "2x+2y", "exp_70": "x^3-1",
    # Imp
    "exp_71": "\\int_0^1 x^2 \\, dx", "exp_72": "\\lim_{x \\to 0} \\frac{\\sin x}{x}", "exp_73": "\\sum_{i=1}^n i^2", "exp_74": "e^{i\\pi} + 1 = 0",
    "exp_75": "\\int x \\, dx", "exp_76": "\\lim_{x \\to \\infty} \\frac{1}{x}", "exp_77": "\\sum_{k=1}^n k", "exp_78": "\\frac{d}{dx} x^2",
    "exp_79": "\\int_0^\\pi \\sin x \\, dx", "exp_80": "\\lim_{x \\to 1} \\frac{x^2-1}{x-1}", "exp_81": "\\sum_{n=1}^\\infty \\frac{1}{n^2}", "exp_82": "\\frac{d}{dx} e^x",
    "exp_83": "\\int \\frac{1}{x} \\, dx", "exp_84": "\\lim_{h \\to 0} \\frac{f(x+h)-f(x)}{h}", "exp_85": "\\prod_{i=1}^n i", "exp_86": "i^2 = -1",
    "exp_87": "\\int_0^1 e^x \\, dx", "exp_88": "\\lim_{x \\to 0} \\frac{1-\\cos x}{x^2}", "exp_89": "\\sum_{i=0}^n \\binom{n}{i}", "exp_90": "\\frac{d}{dx} \\ln x",
    "exp_91": "\\int x e^x \\, dx", "exp_92": "\\lim_{n \\to \\infty} (1+\\frac{1}{n})^n", "exp_93": "\\sum_{i=1}^n \\frac{1}{i}", "exp_94": "\\frac{\\partial f}{\\partial x}",
    "exp_95": "\\int_0^\\infty e^{-x} \\, dx", "exp_96": "\\lim_{x \\to 0} \\frac{\\tan x}{x}", "exp_97": "\\sum_{i=1}^n i^3", "exp_98": "\\nabla \\cdot \\vec{E}",
    "exp_99": "\\int \\sin x \\, dx", "exp_100": "\\lim_{x \\to \\infty} e^{-x}"
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


@app.post("/auth")
async def auth(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    username = data.get("username")
    first_name = data.get("first_name")

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    user = await get_or_create_user(user_id, username, first_name)
    return {"status": "ok", "user_id": user.user_id}


@app.get("/leaderboard")
async def leaderboard(request: Request):
    user_id = request.query_params.get("user_id", "test_user")

    top_players = await get_leaderboard(limit=10)

    user_rank, user_score = await get_user_rank(user_id)

    formatted_top = []
    for idx, (first_name, username, score) in enumerate(top_players):
        name = first_name or username or "Anonymous"
        formatted_top.append({
            "rank": idx + 1,
            "name": name,
            "score": score
        })

    return {
        "top": formatted_top,
        "user_rank": user_rank,
        "user_score": user_score
    }


@app.get("/achievements")
async def get_achievements(request: Request):
    user_id = request.query_params.get("user_id", "test_user")

    total_attempts = await get_total_attempts(user_id)
    avg_score = await get_average_score(user_id)
    high_score = await get_high_score(user_id)
    saved_formulas = await get_saved_formulas(user_id)
    saved_count = len(saved_formulas) if saved_formulas else 0

    unlocked = []

    if total_attempts >= 1: unlocked.append("first_step")
    if total_attempts >= 10: unlocked.append("hard_worker")
    if total_attempts >= 50: unlocked.append("marathon")
    if total_attempts >= 5 and avg_score >= 80: unlocked.append("sniper")
    if high_score >= 3: unlocked.append("on_fire")
    if high_score >= 10: unlocked.append("endless_king")
    if saved_count >= 1: unlocked.append("mathematician")
    if saved_count >= 5: unlocked.append("collector")

    return {"unlocked": unlocked}