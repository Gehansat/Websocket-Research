from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the trained model
model = YOLO('best.pt')  # replace with the path to your trained model

# Create a thread pool
executor = ThreadPoolExecutor(max_workers=4)

def process_image(frame):
    results = model(frame, conf=0.25)
    detected_emotion = "Unknown"
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            c = int(boxes[0].cls.cpu().numpy().item())
            conf = float(boxes[0].conf.cpu().numpy().item())
            detected_emotion = f"{model.names[c]}: {conf:.2f}"
            break
    return detected_emotion

@app.route('/process_image', methods=['POST'])
def handle_process_image():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image in a separate thread
        future = executor.submit(process_image, frame)
        detected_emotion = future.result()

        logging.debug(f"Detected emotion: {detected_emotion}")
        return jsonify({'emotion': detected_emotion})

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500

@app.route('/')
def index():
    return "Real-time Emotion Detection Server"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)