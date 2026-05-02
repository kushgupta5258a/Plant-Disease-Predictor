import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import requests
import io
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Plant Disease Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "model/model.pkl"
LEAF_DETECTOR_PATH = "model/leaf_detector.pkl"
DATA_DIR = "data/dataset/PlantVillage"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# LOAD CLASS NAMES (Hardcoded for deployment)
# -----------------------
@st.cache_resource
def load_class_names():
    # Hardcoded class names from PlantVillage dataset
    class_names = [
        'Pepper__bell___Bacterial_spot',
        'Pepper__bell___healthy',
        'Potato___Early_blight',
        'Potato___healthy',
        'Potato___Late_blight',
        'Tomato__Target_Spot',
        'Tomato__Tomato_mosaic_virus',
        'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Tomato_Bacterial_spot',
        'Tomato_Early_blight',
        'Tomato_healthy',
        'Tomato_Late_blight',
        'Tomato_Leaf_Mold',
        'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite'
    ]
    return class_names, len(class_names)

class_names, num_classes = load_class_names()

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_leaf_detector():
    """Load the leaf detection model (binary classifier: leaf vs non_leaf)"""
    leaf_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    leaf_model.classifier[1] = nn.Linear(leaf_model.classifier[1].in_features, 2)  # Binary classification
    leaf_model.load_state_dict(torch.load(LEAF_DETECTOR_PATH, map_location=DEVICE))
    leaf_model = leaf_model.to(DEVICE)
    leaf_model.eval()
    return leaf_model

model = load_model()

try:
    leaf_detector_model = load_leaf_detector()
    leaf_detector_available = True
except FileNotFoundError:
    leaf_detector_model = None
    leaf_detector_available = False
    st.warning("⚠️ Leaf detector model not found. Please train it first using: python model/train_leaf_detector.py")

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# LEAF DETECTION FUNCTION
# -----------------------
def is_leaf_image(image):
    """Check if the uploaded image is a leaf or not"""
    try:
        image_rgb = image.convert('RGB')
        image_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = leaf_detector_model(image_tensor)
            probabilities = torch.softmax(outputs, 1)
            confidence, predicted = torch.max(probabilities, 1)

        is_leaf = predicted.item() == 0
        leaf_confidence = probabilities[0, 0].item()
        return is_leaf, leaf_confidence
    except Exception as e:
        st.error(f"Error during leaf detection: {e}")
        return None, None

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict(image):
    try:
        image_rgb = image.convert('RGB')
        image_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, 1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Get top 5 predictions
        top_5_probs, top_5_indices = torch.topk(probabilities, 5, 1)
        top_predictions = [(class_names[idx.item()], prob.item()) 
                          for prob, idx in zip(top_5_probs[0], top_5_indices[0])]
        
        return predicted_class, confidence_score, top_predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# -----------------------
# OPENROUTER API FUNCTION
# -----------------------
def get_treatment_advice(disease_name, api_key):
    """Get cure and prevention steps from OpenRouter API"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://plant-disease-predictor.local",
            "X-Title": "Plant Disease Predictor",
            "Content-Type": "application/json"
        }
        
        prompt = f"""You are an agricultural expert. The following plant disease has been detected: {disease_name}

Please provide:
1. **Disease Description**: Brief explanation of the disease
2. **Cure/Treatment Steps**: 3-5 specific steps to treat the disease
3. **Prevention Methods**: 3-5 specific ways to prevent the disease in the future

Format your response clearly with sections and bullet points."""

        data = {
            # ✅ FIX 1: Correct model name for OpenRouter
            "model": "openai/gpt-3.5-turbo",

            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 800   # slightly reduced for faster response
        }
        
        # ✅ FIX 2: Correct endpoint (.ai not .io)
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=15   # ✅ FIX 3: lower timeout to avoid hanging
            # ❌ REMOVED verify=False (was causing SSL + timeout issues)
        )
        
        # Debug (optional but useful)
        # print(response.status_code, response.text)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']

        elif response.status_code == 401:
            return "❌ Invalid API Key. Check your OpenRouter key."

        elif response.status_code == 429:
            return "⏳ Rate limit reached. Try again later."

        else:
            return f"❌ Error {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return "❌ Timeout: API is slow or blocked. Try again or switch network."
    
    except requests.exceptions.ConnectionError:
        return "❌ Connection Error: Check internet or firewall."
    
    except Exception as e:
        return f"❌ Error: {str(e)}"
# -----------------------
# UI LAYOUT
# -----------------------
st.title("🌱 Plant Disease Predictor & Treatment Guide")
st.markdown("---")

# Load API Key from .env
api_key = st.secrets.get("OPENROUTER_API_KEY")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    if api_key and api_key != "your_api_key_here":
        st.success("✅ API Key Loaded from .env")
    else:
        st.warning("⚠️ API Key not configured in .env file")
        st.info("📝 **Setup Instructions**:\n1. Get your API key from https://openrouter.io\n2. Edit the `.env` file\n3. Replace `your_api_key_here` with your actual key\n4. Restart the app")
    st.markdown("[Get OpenRouter API Key](https://openrouter.io/keys)")
    st.markdown("---")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Step 1: Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a plant image",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file:
        st.subheader("🔍 Step 2: Prediction Results")
        
        # First check if image is a leaf
        if leaf_detector_available:
            with st.spinner("🌿 Checking if image is a leaf..."):
                is_leaf, leaf_confidence = is_leaf_image(image)
            
            if is_leaf is None:
                st.error("❌ Error during leaf detection. Please try again.")
            elif not is_leaf:
                st.error(f"❌ This is NOT a leaf image! Leaf confidence: {leaf_confidence*100:.2f}%")
                st.info("📝 Please upload a clear image of a plant leaf for accurate disease prediction.")
            else:
                # It's a leaf, now predict disease
                with st.spinner("🤖 Analyzing disease..."):
                    predicted_disease, confidence, top_predictions = predict(image)
                
                if predicted_disease:
                    st.success(f"✅ Leaf detected! ({leaf_confidence*100:.2f}% confidence)")
                    st.markdown("---")
                    # Display main prediction
                    st.success(f"**Predicted Disease**: {predicted_disease}")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Display top predictions
                    st.markdown("**Top 5 Predictions:**")
                    for i, (disease, prob) in enumerate(top_predictions, 1):
                        st.write(f"{i}. {disease} - {prob*100:.2f}%")
        else:
            st.warning("⚠️ Leaf detector model not available. Skipping leaf validation.")
            with st.spinner("🤖 Analyzing image..."):
                predicted_disease, confidence, top_predictions = predict(image)
            
            if predicted_disease:
                # Display main prediction
                st.success(f"**Predicted Disease**: {predicted_disease}")
                st.metric("Confidence", f"{confidence*100:.2f}%")
                
                # Display top predictions
                st.markdown("**Top 5 Predictions:**")
                for i, (disease, prob) in enumerate(top_predictions, 1):
                    st.write(f"{i}. {disease} - {prob*100:.2f}%")

# Treatment Section
if uploaded_file and 'predicted_disease' in locals() and predicted_disease:
    st.markdown("---")
    st.subheader("💊 Step 3: Treatment & Prevention Guide")
    
    if api_key and api_key != "your_api_key_here":
        if st.button("🔍 Get Treatment Advice", use_container_width=True):
            with st.spinner("⏳ Fetching treatment advice from AI..."):
                advice = get_treatment_advice(predicted_disease, api_key)
            
            st.markdown(advice)
            
            # Option to copy
            st.text_area(
                "Treatment Advice (copyable):",
                value=advice,
                height=300,
                disabled=True
            )
    else:
        st.error("❌ **API Key Not Configured**: Please set up your OpenRouter API key in the `.env` file first")
        st.markdown("""
        **To configure:**
        1. Open `.env` file in your project folder
        2. Replace `your_api_key_here` with your actual OpenRouter API key
        3. Get your free key from [OpenRouter.io](https://openrouter.io/keys)
        4. Save and restart the app
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌾 Plant Disease Predictor v1.0</p>
    <p><small>Powered by EfficientNet-B0 & OpenRouter AI</small></p>
</div>
""", unsafe_allow_html=True)
