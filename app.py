# app.py (for Render) — supports rectangle selection
import os
from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB

def automatic_mask_from_image(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 5)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.bitwise_not(th2)
    mask = cv2.bitwise_or(th, th2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120:
            continue
        if area > 0.9 * (w*h):
            continue
        x,y,ww,hh = cv2.boundingRect(cnt)
        if ww < 8 and hh < 8:
            continue
        cv2.drawContours(final_mask, [cnt], -1, 255, -1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    final_mask = cv2.dilate(final_mask, kernel2, iterations=1)
    return final_mask

@app.route('/')
def home():
    return "Watermark Remover API (rect) running"

@app.route('/remove_with_rect', methods=['POST'])
def remove_with_rect():
    if 'image' not in request.files:
        return jsonify({'error':'no image'}), 400
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img_np = np.array(img)[:, :, ::-1].copy()  # RGB->BGR
    except Exception as e:
        return jsonify({'error':'invalid image'}), 400

    # read coords
    try:
        x = int(request.form.get('x', -1))
        y = int(request.form.get('y', -1))
        w = int(request.form.get('w', -1))
        h = int(request.form.get('h', -1))
    except:
        x = y = w = h = -1

    h_img, w_img = img_np.shape[:2]

    if x == -1 or w <= 0 or h <= 0:
        # auto-detect mask
        mask = automatic_mask_from_image(img_np)
    else:
        # clamp coordinates
        x = max(0, min(w_img-1, x))
        y = max(0, min(h_img-1, y))
        w = max(1, min(w_img - x, w))
        h = max(1, min(h_img - y, h))
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        # create a slightly padded selection to cover soft edges
        pad_x = max(2, int(min(w, h) * 0.08))
        pad_y = pad_x
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w_img, x + w + pad_x)
        y1 = min(h_img, y + h + pad_y)
        mask[y0:y1, x0:x1] = 255
        # optionally refine: try detecting bright pixels within selection to reduce artifacts
        sel_gray = cv2.cvtColor(img_np[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        _, sel_th = cv2.threshold(sel_gray, 220, 255, cv2.THRESH_BINARY)
        # place refined mask back
        refined = np.zeros_like(mask)
        refined[y0:y1, x0:x1] = sel_th
        # if refined non-empty, use it, else use padded rect
        if refined.sum() > 50:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = refined

    # inpaint
    try:
        inpainted = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)
    except Exception as e:
        return jsonify({'error':'inpaint_failed','detail':str(e)}), 500

    out_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(out_rgb)
    buf = BytesIO()
    out_pil.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')
