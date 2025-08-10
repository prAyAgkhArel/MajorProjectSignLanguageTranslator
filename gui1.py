import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load your model and labels as before
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

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        lms = []
        for lm in results.multi_hand_landmarks[0].landmark:
            lms.extend([lm.x, lm.y, lm.z])

        input_data = normalize_landmarks(lms)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]

        input_data = np.array(input_data, dtype=input_dtype).reshape(input_shape)

        interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_idx = np.argmax(output_data)
        predicted_letter = labels[pred_idx]

        # Overlay prediction text on the frame
        cv2.putText(img, f"Predicted: {predicted_letter}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Real-time Sign Language Translator")

webrtc_streamer(key="asl-live", video_frame_callback=video_frame_callback)
