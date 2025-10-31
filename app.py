# app.py
import os
from flask import Flask, request, send_file, jsonify
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

def automatic_mask_from_image(img_bgr):
    """
    Heuristic auto-mask:
    - Convert to gray, enhance contrast (CLAHE)
    - Use adaptive threshold + Canny edges
    - Morphology to join regions
    - Find contours and filter by area and solidity
    - Prefer small/medium blobs and those with higher contrast relative to local background.
    - Optionally boost weight for blobs near corners (many logos are there).
    """
    h, w = img_bgr.shape[:2]
    area_img = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # adaptive threshold for bright/dark marks
    th1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 21, 7)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.bitwise_not(th2)
    # edges
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)

    mask = cv2.bitwise_or(th1, th2)
    mask = cv2.bitwise_or(mask, edges)

    # morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    # contour filter
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(30, 0.0002 * area_img):  # too small
            continue
        if area > 0.06 * area_img:  # too large - probably background
            continue

        x,y,ww,hh = cv2.boundingRect(cnt)
        rect_area = ww*hh
        extent = float(area) / rect_area if rect_area>0 else 0

        # compute mean contrast between contour region and small surrounding band
        pad = int(max(3, min(30, min(ww, hh) * 0.25)))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad); y1 = min(h, y + hh + pad)
        region = gray[y0:y1, x0:x1].astype(np.int32)
        if region.size == 0:
            continue
        # mask local area
        local_mask = np.zeros(region.shape, dtype=np.uint8)
        cnt_rel = cnt - np.array([[x0,y0]])
        cv2.drawContours(local_mask, [cnt_rel], -1, 255, -1)
        inside_vals = region[local_mask==255]
        outside_vals = region[local_mask==0]
        if inside_vals.size==0 or outside_vals.size==0:
            continue
        contrast = float(abs(np.mean(inside_vals) - np.mean(outside_vals)))

        # score heuristics
        score = 0.0
        # area score (prefer small to medium sized)
        score += (1.0 - min(0.5, area / (0.02*area_img)) )  # smaller gets more
        # extent (compact shapes get preference)
        score += extent
        # contrast
        score += min(2.0, contrast/40.0)
        # corner proximity boost
        cx = x + ww/2; cy = y + hh/2
        # distance to nearest corner (normalized)
        corners = [(0,0),(w,0),(0,h),(w,h)]
        min_dist = min([np.hypot(cx-cx0, cy-cy0) for (cx0,cy0) in corners])
        norm_dist = min_dist / np.hypot(w,h)
        corner_boost = 1.0 - norm_dist  # nearer corners get boost
        score += corner_boost * 0.8

        # final threshold for selecting this contour as watermark candidate
        if score >= 1.2:
            cv2.drawContours(final, [cnt], -1, 255, -1)

    # expand/dilate to cover semi-transparent edges
    if final.sum() > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        final = cv2.dilate(final, kernel2, iterations=1)
        # optional smoothing
        final = cv2.medianBlur(final, 5)
    return final

@app.route('/')
def root():
    return "Watermark Remover (auto) running"

@app.route('/remove_auto', methods=['POST'])
def remove_auto():
    if 'image' not in request.files:
        return jsonify({'error': 'no image uploaded'}), 400
    try:
        img = Image.open(request.files['image'].stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'invalid image'}), 400

    img_np = np.array(img)[:, :, ::-1].copy()  # RGB -> BGR
    mask = automatic_mask_from_image(img_np)

    # if nothing detected, fallback: try more aggressive thresholds
    if mask.sum() == 0:
        # second pass: slightly different thresholds (more permissive)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a > max(50, 0.0004 * img_np.shape[0] * img_np.shape[1]) and a < 0.05 * img_np.shape[0] * img_np.shape[1]:
                cv2.drawContours(mask, [c], -1, 255, -1)
        if mask.sum() > 0:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), iterations=1)

    if mask.sum() == 0:
        # nothing found — return original image (client can be notified)
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    try:
        inpainted = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)
    except Exception as e:
        return jsonify({'error': 'inpaint_failed', 'detail': str(e)}), 500

    out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(out)
    buf = BytesIO()
    out_pil.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
