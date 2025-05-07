import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from PIL import Image
import io
import cv2
import time
import threading
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="Rock Paper Scissors Classifier",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #306998;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 2rem;
        font-weight: bold;
        color: #4B8BBE;
        margin-bottom: 0.5rem;
    }
    .prediction-label-invalid {
        font-size: 2rem;
        font-weight: bold;
        color: #E74C3C;
        margin-bottom: 0.5rem;
    }
    .confidence-score {
        font-size: 1.2rem;
        color: #555;
    }
    .team-info {
        background-color: #e6f2ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 2rem;
    }
    .live-prediction {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4B8BBE;
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .invalid-image-box {
        background-color: #fadbd8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    /* Custom CSS to make the camera smaller */
    .stVideoFrame {
        max-width: 480px !important;
        margin: 0 auto;
    }
    /* Making the video frame container smaller */
    .element-container:has(> stVideoFrame) {
        max-width: 480px !important;
        margin: 0 auto;
    }
    .small-camera-container {
        max-width: 480px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Add title and description
st.markdown('<div class="main-header">Rock Paper Scissors Image Classifier</div>', unsafe_allow_html=True)
st.markdown('This app classifies images as rock, paper, or scissors using a CNN model trained on the Rock-Paper-Scissors dataset.')

# Function to load the model
@st.cache_resource
def load_saved_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load class mapping
@st.cache_data
def load_class_mapping(mapping_path):
    try:
        mapping_df = pd.read_csv(mapping_path)
        class_indices = dict(zip(mapping_df['class_name'], mapping_df['class_index']))
        indices_to_class = {v: k for k, v in class_indices.items()}
        return class_indices, indices_to_class
    except Exception as e:
        st.error(f"Error loading class mapping: {e}")
        return {}, {}

# Function to preprocess image
def preprocess_image(img):
    # Resize the image to match the model's expected size
    img = img.resize((150, 150))
    # Convert image to array
    img_array = image.img_to_array(img)
    # Handle RGBA images (4 channels) - convert to RGB (3 channels)
    if img_array.shape[-1] == 4:
        # Convert RGBA to RGB by removing the alpha channel
        img = img.convert('RGB')
        img_array = image.img_to_array(img)
    # Expand dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_array = img_array / 255.0
    return img_array

# Function to make prediction with confidence threshold
def predict(model, img_array, indices_to_class, confidence_threshold=None):
    # Get predictions
    predictions = model.predict(img_array)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])
    # Get the confidence score (probability)
    confidence_score = predictions[0][predicted_class_index]
    
    # Check if the confidence score is below the threshold (only if threshold is provided)
    if confidence_threshold is not None and confidence_score < confidence_threshold:
        return "not_rps", confidence_score, predictions[0]
    
    # Get the class name
    predicted_class = indices_to_class.get(predicted_class_index, "Unknown")
    
    return predicted_class, confidence_score, predictions[0]

# Class for real-time video processing
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, indices_to_class):
        self.model = model
        self.indices_to_class = indices_to_class
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.5  # Make prediction every 0.5 seconds
        self.current_prediction = "Waiting..."
        self.confidence = 0.0
        self.lock = threading.Lock()
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Add prediction info to the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Color based on prediction (green for all predictions)
        color = (0, 255, 0)
        
        text = f"{self.current_prediction} ({self.confidence:.2f})"
        cv2.putText(img, text, (20, 40), font, 1, color, 2, cv2.LINE_AA)
        
        # Make prediction at intervals to reduce processing load
        current_time = time.time()
        if current_time - self.last_prediction_time > self.prediction_interval:
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_img)
            # Preprocess image
            img_array = preprocess_image(pil_img)
            # Make prediction
            try:
                with self.lock:
                    # No confidence threshold for real-time camera
                    predicted_class, confidence_score, _ = predict(
                        self.model, 
                        img_array, 
                        self.indices_to_class
                    )
                    self.current_prediction = predicted_class.upper()
                    self.confidence = confidence_score * 100
            except Exception as e:
                print(f"Error in prediction: {e}")
            
            self.last_prediction_time = current_time
        
        return img

    def get_current_prediction(self):
        with self.lock:
            return self.current_prediction, self.confidence

# Sidebar for model selection and file uploads
with st.sidebar:
    st.markdown('<div class="sub-header">Model Settings</div>', unsafe_allow_html=True)
    
    # Model path selection
    model_path_option = st.selectbox(
        "Choose how to provide the model path:",
        ["Default Path", "Custom Path"]
    )
    
    if model_path_option == "Default Path":
        model_path = "models/best_model.h5"
    else:
        model_path = st.text_input(
            "Enter the path to your .h5 model file:",
            "models/best_model.h5"
        )
    
    # Class mapping path selection
    mapping_path_option = st.selectbox(
        "Choose how to provide the class mapping path:",
        ["Default Path", "Custom Path"]
    )
    
    if mapping_path_option == "Default Path":
        mapping_path = "class_mapping.csv"
    else:
        mapping_path = st.text_input(
            "Enter the path to your class mapping CSV file:",
            "class_mapping.csv"
        )
    
    st.markdown("---")
    st.markdown('<div class="sub-header">Input Options</div>', unsafe_allow_html=True)
    input_option = st.radio(
        "Choose input method:",
        ["Upload Image", "Take Photo", "Real-time Camera Detection"]
    )
    
    # Only show confidence threshold for Upload Image mode
    confidence_threshold = None
    if input_option == "Upload Image":
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.90,
            step=0.05,
            help="Images with confidence scores below this threshold will be classified as 'Not Rock/Paper/Scissors'"
        )

# Load model and class mapping
model = load_saved_model(model_path)
class_indices, indices_to_class = load_class_mapping(mapping_path)

# Main content
if model is None:
    st.warning(f"Please make sure the model file exists at: {model_path}")
elif not class_indices:
    st.warning(f"Please make sure the class mapping file exists at: {mapping_path}")
else:
    st.success("Model and class mapping loaded successfully!")
    
    if input_option == "Upload Image":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Upload Image</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_container_width=True)
                
                # Preprocess image and make prediction
                img_array = preprocess_image(img)
                predicted_class, confidence_score, all_probs = predict(
                    model, 
                    img_array, 
                    indices_to_class,
                    confidence_threshold
                )
                
                with col2:
                    st.markdown('<div class="sub-header">Prediction Result</div>', unsafe_allow_html=True)
                    
                    if predicted_class == "not_rps":
                        # Display for non-RPS images
                        st.markdown(f"""
                        <div class="invalid-image-box">
                            <div class="prediction-label-invalid">Not Rock/Paper/Scissors</div>
                            <div class="confidence-score">
                                The image doesn't appear to be rock, paper, or scissors.<br>
                                Highest confidence: {confidence_score * 100:.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Display for recognized RPS images
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-label">{predicted_class.upper()}</div>
                            <div class="confidence-score">Confidence: {confidence_score * 100:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display all class probabilities
                    st.markdown('<div class="sub-header">Class Probabilities</div>', unsafe_allow_html=True)
                    probs_df = pd.DataFrame({
                        'Class': [indices_to_class.get(i, f"Class {i}") for i in range(len(all_probs))],
                        'Probability': all_probs
                    })
                    probs_df['Probability'] = probs_df['Probability'] * 100
                    
                    # Sort by probability (descending)
                    probs_df = probs_df.sort_values(by='Probability', ascending=False)
                    
                    # Display as bar chart
                    st.bar_chart(probs_df.set_index('Class')['Probability'])
                    
                    # Explanation for non-RPS images
                    if predicted_class == "not_rps":
                        st.info("""
                        The model is designed to recognize rock, paper, and scissors hand gestures.
                        When an image is not clearly one of these gestures or contains other content,
                        it will be classified as "Not Rock/Paper/Scissors".
                        
                        Try uploading a clearer image of a rock, paper, or scissors hand gesture.
                        """)
    
    elif input_option == "Take Photo":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Camera Input</div>', unsafe_allow_html=True)
            # Wrap the camera input in a div with a custom class for styling
            st.markdown('<div class="small-camera-container">', unsafe_allow_html=True)
            camera_img = st.camera_input("Take a picture", key="small_camera")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if camera_img is not None:
                # Process the captured image
                img = Image.open(io.BytesIO(camera_img.getvalue()))
                
                # Preprocess image and make prediction without threshold
                img_array = preprocess_image(img)
                predicted_class, confidence_score, all_probs = predict(
                    model, 
                    img_array, 
                    indices_to_class
                )
                
                with col2:
                    st.markdown('<div class="sub-header">Prediction Result</div>', unsafe_allow_html=True)
                    
                    # Always display the predicted class (no "not_rps" option)
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-label">{predicted_class.upper()}</div>
                        <div class="confidence-score">Confidence: {confidence_score * 100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display all class probabilities
                    st.markdown('<div class="sub-header">Class Probabilities</div>', unsafe_allow_html=True)
                    probs_df = pd.DataFrame({
                        'Class': [indices_to_class.get(i, f"Class {i}") for i in range(len(all_probs))],
                        'Probability': all_probs
                    })
                    probs_df['Probability'] = probs_df['Probability'] * 100
                    
                    # Sort by probability (descending)
                    probs_df = probs_df.sort_values(by='Probability', ascending=False)
                    
                    # Display as bar chart
                    st.bar_chart(probs_df.set_index('Class')['Probability'])
    
    elif input_option == "Real-time Camera Detection":
        st.markdown('<div class="sub-header">Real-time Hand Gesture Recognition</div>', unsafe_allow_html=True)
        st.markdown("""
        Show your hand gesture (rock, paper, or scissors) to the camera. 
        The model will predict your gesture in real-time.
        """)
        
        # Create two columns for better layout - camera on left, predictions on right
        cam_col, pred_col = st.columns([1, 1])
        
        with cam_col:
            # RTC Configuration with multiple free STUN servers for better reliability
            rtc_config = RTCConfiguration(
                {"iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun.ekiga.net"]}
                ]}
            )
            
            # Custom VideoTransformer with the model
            transformer = VideoTransformer(model, indices_to_class)
            
            # Start the WebRTC streamer with smaller video size and lower framerate
            webrtc_ctx = webrtc_streamer(
                key="rock-paper-scissors",
                video_transformer_factory=lambda: transformer,
                rtc_configuration=rtc_config,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 320},
                        "height": {"ideal": 240},
                        "frameRate": {"max": 15}
                    },
                    "audio": False
                },
                async_processing=True,
                video_html_attrs={
                    "style": {"width": "100%", "max-width": "320px", "margin": "0 auto", "display": "block"},
                    "autoPlay": True,
                    "controls": False,
                },
            )
        
        with pred_col:
            # Display troubleshooting tips
            expander = st.expander("Camera Not Working? Try These Tips", expanded=False)
            with expander:
                st.markdown("""
                1. **Check Browser Permissions**: Make sure your browser has permission to use the camera
                2. **Try a Different Browser**: Chrome or Firefox often work best
                3. **Close Other Applications**: Make sure no other applications are using your camera
                4. **Refresh the Page**: Sometimes a simple refresh resolves connection issues
                5. **Try the 'Take Photo' option**: If real-time detection doesn't work, the single-photo option might
                """)
            
            # Create a placeholder for prediction display
            prediction_placeholder = st.empty()
            
            # Add visual representation of rock, paper, scissors
            st.markdown("""
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <div style="text-align: center;">
                    <div style="font-size: 40px;">✊</div>
                    <div>Rock</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 40px;">✋</div>
                    <div>Paper</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 40px;">✌️</div>
                    <div>Scissors</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Update prediction display while the stream is running (safer implementation)
            if webrtc_ctx.state.playing:
                try:
                    # Using session state to store the stop flag
                    if 'stop_thread' not in st.session_state:
                        st.session_state.stop_thread = False
                    
                    prediction_placeholder.markdown("""
                    <div class="live-prediction">
                        Waiting for camera...
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Use a safer approach with Streamlit's built-in functions
                    for _ in range(100):  # Limit iterations for safety
                        if not webrtc_ctx.state.playing or st.session_state.stop_thread:
                            break
                        
                        predicted_class, confidence = transformer.get_current_prediction()
                        
                        prediction_placeholder.markdown(f"""
                        <div class="live-prediction">
                            Current Prediction: {predicted_class} ({confidence:.2f}%)
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Sleep to avoid UI lag
                        time.sleep(0.5)
                except Exception as e:
                    st.error(f"Error in real-time prediction: {e}")
                    st.session_state.stop_thread = True
            else:
                prediction_placeholder.markdown("""
                <div class="live-prediction">
                    Start the camera to see predictions
                </div>
                """, unsafe_allow_html=True)

# Instructions section
st.markdown('<div class="sub-header">How to Use</div>', unsafe_allow_html=True)
st.markdown("""
1. Make sure the model and class mapping files are correctly loaded
2. Choose your preferred input method:
   - **Upload Image**: Upload an image of rock, paper, or scissors (includes confidence threshold option)
   - **Take Photo**: Take a picture using your device's camera
   - **Real-time Camera Detection**: Show hand gestures to your camera in real-time

**Note:** For best results, ensure that:
- The hand gesture is clearly visible and centered in the frame
- There is good lighting in the environment
- Your hand is positioned against a clean, contrasting background
""")

# Team information
st.markdown('<div class="sub-header">Team Information</div>', unsafe_allow_html=True)
st.markdown("""
<div class="team-info">
    <b>Team Members:</b><br>
    1. Jihan Nabilah - [2208107010035] - [Team Leader]<br>
    2. Shofia Nurul Huda - [2208107010015]<br>
    3. Farhanul Khair - [2208107010076]<br>
    4. M. Bintang Indra Hidayat - [2208107010023]<br>
    5. Ahmad Syah Ramadhan - [2208107010033]<br>
    <br>
    <b>Project:</b> UAS Deployment Rock Paper Scissors Image Classification using CNN<br>
    <b>Course:</b> Praktikum Pembelajaran Mesin A <br>
</div>
""", unsafe_allow_html=True)