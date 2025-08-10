import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load TFLite model (adjust path if needed)
interpreter = tf.lite.Interpreter(model_path="asl_letters.tflite")
interpreter.allocate_tensors()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Normalization function (as per your code)
def normalize_landmarks(lms):
    WRIST = 0
    MCP_MIDDLE = 9
    pts = np.array(lms).reshape(21, 3)
    wrist = pts[WRIST].copy()
    pts -= wrist
    ref_vec = pts[MCP_MIDDLE]
    scale = np.linalg.norm(ref_vec)
    if scale < 1e-6:
        scale = np.linalg.norm(pts).mean() + 1e-6
    pts /= scale
    return pts.flatten().astype(np.float32)

st.title("Sign Language Translator Web App")

img_file_buffer = st.camera_input("Show your hand")

if img_file_buffer is not None:
    # Read image as numpy array
    image = np.array(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe Hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        lms = []
        for lm in landmarks.landmark:
            lms.extend([lm.x, lm.y, lm.z])

        # Normalize landmarks to match model input
        input_data = normalize_landmarks(lms)

        # Get input details from interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Model expects input shape and dtype
        input_shape = input_details[0]['shape']       # e.g., [1, 63]
        input_dtype = input_details[0]['dtype']       # usually np.float32

        # Reshape and typecast input data accordingly
        input_data = np.array(input_data, dtype=input_dtype).reshape(input_shape)

        # Resize tensor if dynamic shape
        if not np.array_equal(input_shape, input_data.shape):
            interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
            interpreter.allocate_tensors()

        # Set tensor and invoke interpreter
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output and find predicted label index
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_idx = np.argmax(output_data)
        predicted_letter = labels[pred_idx]

        st.write(f"Predicted Letter: **{predicted_letter}**")

        # Optionally display the captured image
        st.image(image_rgb, caption="Captured Image", use_column_width=True)
    else:
        st.write("No hand detected. Please show your hand clearly in the camera.")
