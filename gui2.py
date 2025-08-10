import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import threading
import queue
import time

## --- TTS Worker Setup ---
tts_engine = pyttsx3.init()
speech_queue = queue.Queue()
speech_lock = threading.Lock()

def speech_worker():
    while True:
        letter = speech_queue.get()
        try:
            with speech_lock:
                tts_engine.say(letter)
                tts_engine.runAndWait()
            time.sleep(0.2)
        except Exception as e:
            print(f"[Speech Worker] ERROR: {e}")
        speech_queue.task_done()
threading.Thread(target=speech_worker, daemon=True).start()
def enqueue_speech(letter):
    if letter:
        speech_queue.put(letter)

## --- Model & MediaPipe setup ---
interpreter = tf.lite.Interpreter(model_path="asl_letters.tflite")
interpreter.allocate_tensors()
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
def normalize_landmarks(lms):
    WRIST, MCP_MIDDLE = 0, 9
    pts = np.array(lms).reshape(21, 3)
    wrist = pts[WRIST].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[MCP_MIDDLE])
    if scale < 1e-6:
        scale = np.linalg.norm(pts).mean() + 1e-6
    pts /= scale
    return pts.flatten().astype(np.float32)

last_predicted = None

def is_landmarks_valid(lms):
    """Reject if bounding box too small/large, or geometric checks fail."""
    xs = [x for i, x in enumerate(lms) if i % 3 == 0]
    ys = [y for i, y in enumerate(lms) if i % 3 == 1]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    box_area = (max_x - min_x) * (max_y - min_y)
    # Typical normalized hand area range; tune these as needed!
    if box_area < 0.01 or box_area > 0.8:  # Too small/large? (0.01-0.8 are loose example thresholds)
        return False
    return True

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global last_predicted
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    valid_hand = False
    predicted_letter = None

    if results.multi_hand_landmarks and results.multi_handedness:
        lms_obj = results.multi_hand_landmarks[0]
        lms = []
        for lm in lms_obj.landmark:
            lms.extend([lm.x, lm.y, lm.z])

        # --- Handedness/confidence filter ---
        handedness = results.multi_handedness[0].classification[0]
        score = handedness.score
        if score < 0.5:  # Discard hand detections lower than 0.5 confidence
            cv2.putText(img, "Hand detection too weak!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        elif not is_landmarks_valid(lms):  # Area filter for dummy shapes
            cv2.putText(img, "Invalid hand shape!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            # Passed filters: Do prediction
            valid_hand = True
            input_data = normalize_landmarks(lms)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_shape = input_details[0]["shape"]
            input_dtype = input_details[0]["dtype"]
            input_data = np.array(input_data, dtype=input_dtype).reshape(input_shape)
            if not np.array_equal(input_shape, input_data.shape):
                interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
                interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred_idx = np.argmax(output_data)
            conf = output_data[0][pred_idx]
            # --- Prediction confidence filter ---
            if conf < 0.8:  # Too low? Reject.
                cv2.putText(img, "Low prediction confidence!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            else:
                predicted_letter = labels[pred_idx]
                cv2.putText(img, f"Predicted: {predicted_letter} ({conf:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                # Speak only on new predictions
                if predicted_letter != last_predicted:
                    enqueue_speech(predicted_letter)
                    last_predicted = predicted_letter
    else:
        cv2.putText(img, "No hand detected", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Sign Language Translator with Dummy Hand Filtering")
webrtc_streamer(key="asl-live", video_frame_callback=video_frame_callback)
