from flask import Flask
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
import logging
import mediapipe as mp
from ultralytics import YOLO
import joblib
import json

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the trained models
emotion_model = YOLO('best.pt')
gesture_model = joblib.load('gesture.joblib')

# Set up MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_features(hand_landmarks):
    palm_center = np.mean([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark[:5]], axis=0)
    fingers = [hand_landmarks.landmark[tip] for tip in [8, 12, 16, 20]]
    features = []
    for finger in fingers:
        features.extend([finger.x - palm_center[0], finger.y - palm_center[1], finger.z - palm_center[2]])
    return features

def process_image(frame):
    # Emotion detection
    results = emotion_model(frame, conf=0.25)
    detected_emotion = "Unknown"
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            c = int(boxes[0].cls.cpu().numpy().item())
            conf = float(boxes[0].conf.cpu().numpy().item())
            detected_emotion = f"{emotion_model.names[c]}: {conf:.2f}"
            break

    # Gesture recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    detected_gesture = "Unknown"
    
    if results.multi_hand_landmarks:
        all_hand_features = []
        for hand_landmarks in results.multi_hand_landmarks:
            features = extract_features(hand_landmarks)
            all_hand_features.extend(features)
        if len(all_hand_features) < 24:
            all_hand_features.extend([0] * (24 - len(all_hand_features)))
        probabilities = gesture_model.predict_proba([all_hand_features[:24]])[0]
        prediction = gesture_model.predict([all_hand_features[:24]])[0]
        confidence_threshold = 0.7
        if probabilities[prediction] >= confidence_threshold:
            detected_gesture = f"Crying: {probabilities[prediction]:.2f}" if prediction == 1 else f"Clapping: {probabilities[prediction]:.2f}"

    return detected_emotion, detected_gesture

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('stream_frame')
def handle_stream_frame(data):
    try:
        # Decode and process the image
        image_data = data['image']
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the image
        detected_emotion, detected_gesture = process_image(frame)
        
        # Emit the results back to the client
        socketio.emit('detection_result', {
            'emotion': detected_emotion,
            'gesture': detected_gesture
        })
        
    except Exception as e:
        logging.error(f"Error in processing stream: {str(e)}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)