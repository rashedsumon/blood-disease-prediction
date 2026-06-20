import streamlit as st
import os
from PIL import Image
import torch
from data_loader import download_and_prepare_dataset
from model import BloodCellCNN, predict_image

# Page Configurations
st.set_page_config(page_title="Blood Cell Anomaly Detector", page_icon="🩸", layout="centered")

st.title("🩸 Blood Cell Anomaly Detection AI")
st.write("Identify cellular abnormalities instantly using deep learning.")

# --- SECTION 1: DATASET INITIALIZATION ---
st.subheader("1. Dataset Status")
with st.spinner("Initializing dataset from Kaggle via kagglehub..."):
    try:
        dataset_path = download_and_prepare_dataset()
        st.success(f"Dataset securely linked! Ready for analysis.")
        with st.expander("Show Local Data Path Details"):
            st.code(dataset_path)
    except Exception as e:
        st.error(f"Failed to fetch dataset: {e}")

# --- SECTION 2: AI MODEL INITIALIZATION ---
# We initialize a mock/untrained model structure for deployment presentation.
# In a real environment, you would load pre-saved weights (.pth file).
@st.cache_resource
def load_ai_model():
    model = BloodCellCNN(num_classes=2)
    # model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
    return model

model = load_ai_model()

# --- SECTION 3: IMAGE UPLOAD & INFERENCE ---
st.write("---")
st.subheader("2. Analyze Patient Blood Smear Sample")
uploaded_file = st.file_uploader("Upload a microscopic cell image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Patient Sample Image", use_column_width=True)
    
    # Save image temporarily for processing
    temp_path = "temp_sample_image.png"
    image.save(temp_path)
    
    st.write("🤖 **AI Analysis running...**")
    
    # Run prediction logic from model.py
    label, confidence = predict_image(temp_path, model)
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    # Display results with dynamic styling
    st.write("### Analysis Diagnostics:")
    if label == "Anomaly":
        st.error(f"⚠️ **Result:** Anomaly Detected")
    else:
        st.success(f"✅ **Result:** Normal Cell Structure")
        
    st.metric(label="Confidence Level", value=f"{confidence * 100:.2f}%")