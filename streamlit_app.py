import streamlit as st
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN
from deepface import DeepFace
import pickle

# Load trained KNN model
with open("data/embeddings/knn_classifier.pkl", "rb") as f:
    knn_model, X_train, y_train = pickle.load(f)

# Initialize MTCNN detector
detector = MTCNN()

# Streamlit UI Setup
st.set_page_config(page_title="AeroFace - Face Recognition", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 AeroFace - Face Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image with one or more faces to recognize them.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and convert image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Detecting and recognizing faces..."):
        results = detector.detect_faces(img_rgb)

        if not results:
            st.error("❌ No face detected in the image.")
        else:
            img_annotated = img_rgb.copy()

            for result in results:
                x, y, w, h = result["box"]
                x, y = max(0, x), max(0, y)
                face = img_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (160, 160))

                try:
                    # Get embedding
                    embedding = DeepFace.represent(face_resized, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    embedding = np.array(embedding).reshape(1, -1)

                    # Predict
                    prediction = knn_model.predict(embedding)[0]
                    confidence = knn_model.predict_proba(embedding).max()

                    # Label (Always show prediction — no "Unknown" logic)
                    label = f"{prediction} ({confidence*100:.1f}%)"
                    color = (0, 255, 0)  # Green box

                    # Draw
                    cv2.rectangle(img_annotated, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(img_annotated, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    st.warning(f"⚠️ Error processing a face: {e}")

            st.success(f"✅ Detected {len(results)} face(s)")
            st.image(img_annotated, caption="🎯 Recognition Result", use_column_width=True)
