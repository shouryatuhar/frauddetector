"""
Flask web demo for the Screen Recapture Detector.

Loads the trained TensorFlow model ONCE at startup by importing from predict.py,
then serves a web UI and a /predict endpoint.
"""

import os
import time
import tempfile
from flask import Flask, request, jsonify, render_template

# ─── Import the existing prediction pipeline (model loads at import time) ───
from predict import predict as run_prediction

app = Flask(__name__)

# Max upload size: 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use JPG, PNG, or WebP.'}), 400

    # Save to a temp file (predict.py expects a file path)
    suffix = '.' + file.filename.rsplit('.', 1)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()

        start = time.time()
        probability = run_prediction(tmp.name)
        elapsed_ms = (time.time() - start) * 1000.0
    finally:
        os.unlink(tmp.name)

    label = 'SCREEN' if probability >= 0.5 else 'REAL'

    return jsonify({
        'probability': round(probability, 6),
        'label': label,
        'inference_ms': round(elapsed_ms, 1),
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
