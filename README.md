# 🧠 AeroFace - Face Recognition System

A comprehensive face recognition web application built with Streamlit, MTCNN, and DeepFace. AeroFace can detect and recognize faces in uploaded images using state-of-the-art deep learning models.

## � Live Demo

Try the application online: **[AeroFace Live App](https://aeroface-87wljkmrufzyvvbhsc4apd.streamlit.app/)**

## �🌟 Features

- **Real-time Face Detection**: Uses MTCNN for accurate face detection
- **Face Recognition**: Powered by DeepFace with FaceNet embeddings
- **Multiple Face Support**: Can detect and recognize multiple faces in a single image
- **Web Interface**: User-friendly Streamlit web application
- **Confidence Scoring**: Shows recognition confidence percentages
- **Visual Feedback**: Displays bounding boxes and labels on detected faces

## 🏗️ Project Structure

```
AeroFace/
├── app.py                          # Basic Streamlit app (single face)
├── streamlit_app.py               # Enhanced Streamlit app (multiple faces)
├── requirements.txt               # Python dependencies
├── data/
│   ├── lfw-funneled/             # Original LFW dataset
│   ├── processed_faces_mtcnn/    # MTCNN processed faces
│   └── embeddings/
│       └── knn_classifier.pkl    # Trained KNN model
└── notebooks/                     # Development notebooks
    ├── Week1_Data_Extraction.ipynb
    ├── Week2_Face_Detection_MTCNN.ipynb
    ├── Week3_Face_Embedding_and_Classifier.ipynb
    ├── Week4_Face_Recognition_from_Image.ipynb
    └── Week6_Analysis_and_Improvements.ipynb
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Devb1234/AeroFace.git
   cd AeroFace
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   
   For the enhanced version (recommended):
   ```bash
   streamlit run streamlit_app.py
   ```
   
   For the basic version:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Upload an image and start recognizing faces!

## 🔧 Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Face Detection** | MTCNN |
| **Face Recognition** | DeepFace + FaceNet |
| **Classification** | K-Nearest Neighbors (KNN) |
| **Image Processing** | OpenCV, PIL |
| **Data Processing** | NumPy, scikit-learn |

## 📊 Model Details

- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Embedding**: FaceNet (128-dimensional embeddings)
- **Classification**: K-Nearest Neighbors with trained embeddings
- **Dataset**: LFW (Labeled Faces in the Wild) dataset

## 🎯 Usage

1. **Launch the application** using one of the Streamlit commands above
2. **Upload an image** (JPG, JPEG, or PNG format)
3. **Wait for processing** - the app will:
   - Detect faces in the image
   - Extract face embeddings
   - Classify each face using the trained model
4. **View results** with bounding boxes and confidence scores

## 📝 Development Workflow

The project follows a structured development approach documented in Jupyter notebooks:

1. **Week 1**: Data extraction and preprocessing from LFW dataset
2. **Week 2**: Face detection implementation using MTCNN
3. **Week 3**: Face embedding generation and KNN classifier training
4. **Week 4**: Complete face recognition pipeline from images
5. **Week 6**: Analysis, improvements, and optimization

## 🔍 Key Components

### Face Detection (MTCNN)
- Detects faces with high accuracy
- Provides bounding box coordinates
- Handles multiple faces in single image

### Face Recognition (DeepFace + FaceNet)
- Generates 128-dimensional face embeddings
- Uses pre-trained FaceNet model
- Robust to lighting and pose variations

### Classification (KNN)
- Trained on processed face embeddings
- Provides confidence scores
- Fast inference for real-time applications

## 📋 Dependencies

```
streamlit==1.35.0
opencv-python-headless==4.9.0.80
numpy==1.23.5                # Downgraded to be compatible with TensorFlow 2.12.0
pillow==10.2.0
mtcnn==0.1.1
deepface==0.0.79
scikit-learn==1.3.2
tensorflow==2.12.0
protobuf==4.25.3             # Compatible and safe
```

## 🎨 Application Versions

### Basic App (`app.py`)
- Single face recognition
- Simple interface
- Basic error handling

### Enhanced App (`streamlit_app.py`)
- Multiple face recognition
- Improved UI with styling
- Better error handling and user feedback
- Progress indicators

## 🚨 Troubleshooting

### Common Issues

1. **No face detected**
   - Ensure the image contains clear, visible faces
   - Try images with better lighting
   - Face should be reasonably sized in the image

2. **Recognition errors**
   - Check if the person is in the training dataset
   - Verify image quality and resolution
   - Ensure face is not heavily occluded

3. **Installation issues**
   - Update pip: `pip install --upgrade pip`
   - Install dependencies one by one if batch installation fails
   - Check Python version compatibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LFW Dataset**: Labeled Faces in the Wild dataset for training data
- **MTCNN**: Multi-task Cascaded Convolutional Networks for face detection
- **DeepFace**: Deep learning face recognition library
- **FaceNet**: Face recognition model architecture
- **Streamlit**: For the amazing web app framework

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the development notebooks for detailed implementation
3. Open an issue in the repository

---

**Made with ❤️ and deep learning**
