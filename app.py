from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

@app.route('/remove_watermark', methods=['POST'])
def remove_watermark():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    img_array = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Simple inpainting mask (detect bright text areas)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp.name, result)

    return send_file(temp.name, mimetype='image/png')

@app.route('/')
def home():
    return "✅ Watermark Remover API is Running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
