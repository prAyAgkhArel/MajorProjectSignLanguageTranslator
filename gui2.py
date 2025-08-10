import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# --- Model & MediaPipe setup ---
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

def is_landmarks_valid(lms):
    """Reject if bounding box too small/large, or geometric checks fail."""
    xs = [x for i, x in enumerate(lms) if i % 3 == 0]
    ys = [y for i, y in enumerate(lms) if i % 3 == 1]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    box_area = (max_x - min_x) * (max_y - min_y)
    # Typical normalized hand area range; tune these as needed
    if box_area < 0.01 or box_area > 0.8:  # Adjust thresholds as necessary
        return False
    return True

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        lms_obj = results.multi_hand_landmarks[0]
        lms = []
        for lm in lms_obj.landmark:
            lms.extend([lm.x, lm.y, lm.z])

        # Filter by MediaPipe handedness confidence
        handedness = results.multi_handedness[0].classification[0]
        score = handedness.score
        if score < 0.5:
            cv2.putText(img, "Hand detection too weak!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        elif not is_landmarks_valid(lms):
            cv2.putText(img, "Invalid hand shape!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
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
            if conf < 0.8:
                cv2.putText(img, "Low prediction confidence!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            else:
                predicted_letter = labels[pred_idx]
                cv2.putText(img, f"Predicted: {predicted_letter} ({conf:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(img, "No hand detected", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Sign Language Translator with Confidence Filtering ")

webrtc_streamer(
    key="asl-live",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}  # Disable microphone capture
)