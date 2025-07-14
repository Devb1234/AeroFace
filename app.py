import streamlit as st
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN
from deepface import DeepFace
import pickle
import io

# Load trained KNN model
with open("data/embeddings/knn_classifier.pkl", "rb") as f:
    knn_model, X_train, y_train = pickle.load(f)

# Initialize MTCNN detector
detector = MTCNN()

st.set_page_config(page_title="AeroFace Recognition App", layout="centered")
st.title("🧠 AeroFace - Face Recognition Web App")
st.markdown("Upload an image of a known person to identify them.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as RGB
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Show uploaded image
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # Detect face
    results = detector.detect_faces(img_rgb)

    if not results:
        st.error("❌ No face detected in the image.")
    else:
        result = results[0]
        x, y, w, h = result['box']
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
            label = f"{prediction} ({confidence*100:.2f}%)"

            # Draw result
            img_annotated = img_rgb.copy()
            cv2.rectangle(img_annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Show result
            st.image(img_annotated, caption="🎯 Recognition Result", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Face recognition failed: {e}")
