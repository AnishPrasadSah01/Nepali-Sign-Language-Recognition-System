# ============================================================
# NSL RECOGNITION SYSTEM - COMPLETE STREAMLIT APP
# T6: Nepali Sign Language to Text/Speech Translation
# Team: Anish, Ichingsha, Bibek, Alex
# ============================================================

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import time
from gtts import gTTS
import os
from io import BytesIO
import tempfile
import pandas as pd
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NSL Recognition System - T6",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #dc3545 0%, #ff6b6b 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.8rem; margin: 0; }
    .main-header p { font-size: 1.2rem; margin: 0; opacity: 0.9; }
    
    .feature-tag {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 3px;
    }
    
    .prediction-box {
        background: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 3px solid #28a745;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-text {
        font-size: 4.5rem;
        font-weight: bold;
        color: #28a745;
        margin: 10px 0;
    }
    .confidence-text {
        font-size: 1.2rem;
        color: #666;
    }
    .model-text {
        font-size: 0.9rem;
        color: #999;
    }
    
    .history-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
    }
    .history-item {
        padding: 8px 12px;
        margin: 4px 0;
        background: white;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #dc3545;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if 'running' not in st.session_state:
    st.session_state.running = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = "Ready"
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0
if 'model_type' not in st.session_state:
    st.session_state.model_type = "MobileNetV2"

# ============================================================
# NSL CLASSES
# ============================================================

NSL_CLASSES = [
    'KA', 'KHA', 'GA', 'GHA', 'NGA', 'CHA', 'CHHA', 'JA', 'JHA', 'YAN',
    'TA', 'THA', 'DA', 'DHA', 'NA', 'TAA', 'THAA', 'DAA', 'DHAA', 'NAA',
    'PA', 'PHA', 'BA', 'BHA', 'MA', 'YA', 'RA', 'LA', 'WA', 'T_SHA',
    'M_SHA', 'D_SHA', 'HA', 'KSHA', 'TRA', 'GYA'
]

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    """Load both MobileNetV2 and LSTM models"""
    models = {"mobilenetv2": None, "lstm": None}
    
    # Try multiple paths for MobileNetV2
    mobilenetv2_paths = [
        "models/mobilenetv2_nsl_final.h5",
        "models/mobilenetv2_nsl_final.keras",
        "mobilenetv2_nsl_final.h5",
    ]
    
    for path in mobilenetv2_paths:
        if os.path.exists(path):
            try:
                models["mobilenetv2"] = keras.models.load_model(path)
                break
            except Exception as e:
                continue
    
    # Try multiple paths for LSTM
    lstm_paths = [
        "models/lstm_architecture.h5",
        "lstm_architecture.h5",
    ]
    
    for path in lstm_paths:
        if os.path.exists(path):
            try:
                models["lstm"] = keras.models.load_model(path)
                break
            except Exception as e:
                continue
    
    return models

# Load models
with st.spinner("Loading models..."):
    models = load_models()

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return hands, mp_hands

hands, mp_hands = init_mediapipe()
mp_drawing = mp.solutions.drawing_utils

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_frame(frame):
    """Preprocess frame for MobileNetV2 input"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)

def extract_landmarks(frame):
    """Extract hand landmarks from a frame"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        while len(landmarks) < 63:
            landmarks.extend([0, 0, 0])
        
        return np.array(landmarks[:63])
    
    return np.zeros(63)

def predict_mobilenetv2(frame, model, threshold=0.7):
    """Predict using MobileNetV2"""
    if model is None:
        return "Model Not Loaded", 0.0
    
    try:
        processed = preprocess_frame(frame)
        predictions = model.predict(processed, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        if confidence >= threshold and pred_class < len(NSL_CLASSES):
            return NSL_CLASSES[pred_class], confidence
        else:
            return "Low Confidence", confidence
    except Exception as e:
        return "Error", 0.0

def predict_lstm(frame, model, threshold=0.7):
    """Predict using LSTM with frame sequence"""
    if model is None:
        return "LSTM Not Available", 0.0
    
    try:
        landmarks = extract_landmarks(frame)
        sequence = np.tile(landmarks, (30, 1))
        sequence = sequence.reshape(1, 30, 63)
        
        predictions = model.predict(sequence, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        if confidence >= threshold and pred_class < len(NSL_CLASSES):
            return NSL_CLASSES[pred_class], confidence
        else:
            return "Low Confidence", confidence
    except Exception as e:
        return "Error", 0.0

def text_to_speech(text, lang='ne'):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        return None

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🤟 Nepali Sign Language Recognition</h1>
    <p>T6: Real-time Translation with Text-to-Speech</p>
    <div>
        <span class="feature-tag">📷 Video Frame Processing</span>
        <span class="feature-tag">🔊 Text-to-Speech</span>
        <span class="feature-tag">🧠 Sequence Modeling (LSTM)</span>
        <span class="feature-tag">📊 36 NSL Classes</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 👥 Team Members")
    st.markdown("""
    - **Anish** 👑 (Team Lead)
    - **Ichingsha** 🔬 (Research Lead)
    - **Bibek** 💻 (CV/NLP Engineer)
    - **Alex** 🎨 (Integration Lead)
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    model_choice = st.selectbox(
        "Select Model",
        ["MobileNetV2 (Baseline)", "LSTM (Advanced)"],
        index=0
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        0.5, 0.95, 0.7, 0.05
    )
    
    tts_lang = st.selectbox(
        "TTS Language",
        ["Nepali (नेपाली)", "English"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    
    if models["mobilenetv2"] is not None:
        st.success("✅ MobileNetV2: Loaded")
    else:
        st.error("❌ MobileNetV2: Not Found")
    
    if models["lstm"] is not None:
        st.success("✅ LSTM: Loaded")
    else:
        st.warning("⚠️ LSTM: Not Found")
    
    st.markdown("---")
    st.markdown("### 📊 Performance")
    st.metric("Baseline Accuracy", "99.9167%")
    st.metric("Classes", "36")
    st.metric("Val Samples", "7,200")

# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["📷 Live Camera", "📹 Video Upload", "📊 Results"])

# ============================================================
# TAB 1: LIVE CAMERA
# ============================================================

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📷 Live Camera")
        camera_placeholder = st.empty()
        
        col_controls = st.columns(3)
        with col_controls[0]:
            start_btn = st.button("▶️ Start", use_container_width=True, type="primary")
            if start_btn:
                st.session_state.running = True
        with col_controls[1]:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)
            if stop_btn:
                st.session_state.running = False
        with col_controls[2]:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
            if clear_btn:
                st.session_state.history = []
                st.session_state.current_prediction = "Ready"
        
        if st.session_state.running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Camera not accessible!")
                st.session_state.running = False
            else:
                last_pred_time = time.time()
                
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, 
                                hand_landmarks, 
                                mp_hands.HAND_CONNECTIONS
                            )
                    
                    if time.time() - last_pred_time > 0.5:
                        if model_choice == "MobileNetV2 (Baseline)":
                            pred, conf = predict_mobilenetv2(frame, models["mobilenetv2"], confidence_threshold)
                            st.session_state.model_type = "MobileNetV2"
                        else:
                            pred, conf = predict_lstm(frame, models["lstm"], confidence_threshold)
                            st.session_state.model_type = "LSTM"
                        
                        st.session_state.current_prediction = pred
                        st.session_state.current_confidence = conf
                        
                        if pred not in ["Ready", "Low Confidence", "Error", "Model Not Loaded", "LSTM Not Available"]:
                            st.session_state.history.append(pred)
                            if len(st.session_state.history) > 20:
                                st.session_state.history = st.session_state.history[-20:]
                        
                        last_pred_time = time.time()
                    
                    if st.session_state.current_prediction not in ["Ready", "Low Confidence", "Error", "Model Not Loaded", "LSTM Not Available"]:
                        cv2.putText(
                            frame, 
                            f"{st.session_state.current_prediction} ({st.session_state.current_confidence:.1%})",
                            (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.2, 
                            (0, 255, 0), 
                            3
                        )
                        cv2.putText(
                            frame, 
                            f"Model: {st.session_state.model_type}",
                            (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (255, 255, 255), 
                            2
                        )
                    
                    camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                cap.release()
    
    with col2:
        st.subheader("🎯 Recognition Result")
        
        if st.session_state.current_prediction not in ["Ready", "Low Confidence", "Error", "Model Not Loaded", "LSTM Not Available"]:
            st.markdown(f"""
            <div class="prediction-box">
                <p style="font-size:1.2rem; color:#666;">Recognized Sign</p>
                <p class="prediction-text">{st.session_state.current_prediction}</p>
                <p class="confidence-text">Confidence: {st.session_state.current_confidence:.1%}</p>
                <p class="model-text">Model: {st.session_state.model_type}</p>
            </div>
            """, unsafe_allow_html=True)
            
            lang_code = 'ne' if tts_lang == "Nepali (नेपाली)" else 'en'
            if st.button("🔊 Speak", use_container_width=True):
                with st.spinner("Generating speech..."):
                    audio = text_to_speech(st.session_state.current_prediction, lang=lang_code)
                    if audio:
                        st.audio(audio, format='audio/mp3')
                        st.success(f"🔊 Speaking in {tts_lang}")
                    else:
                        st.error("❌ TTS failed. Please check internet.")
        else:
            st.info("👆 Show a sign to the camera")
        
        st.subheader("📝 History")
        st.markdown('<div class="history-box">', unsafe_allow_html=True)
        if st.session_state.history:
            for pred in st.session_state.history[-10:]:
                st.markdown(f'<div class="history-item">{pred}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#999; text-align:center;">No predictions yet</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2: VIDEO UPLOAD
# ============================================================

with tab2:
    st.subheader("📹 Upload Video for Processing")
    st.markdown("Process video frames with **Video Frame Processing** and **Sequence Modeling**")
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        if st.button("▶️ Process Video", use_container_width=True):
            with st.spinner("Processing video frames..."):
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                predictions = []
                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % 10 == 0:
                        if model_choice == "MobileNetV2 (Baseline)":
                            pred, conf = predict_mobilenetv2(frame, models["mobilenetv2"], confidence_threshold)
                        else:
                            pred, conf = predict_lstm(frame, models["lstm"], confidence_threshold)
                        
                        predictions.append((frame_count, pred, conf))
                        
                        display_frame = frame.copy()
                        if pred not in ["Low Confidence", "Error"]:
                            cv2.putText(
                                display_frame, 
                                f"{pred} ({conf:.1%})", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (0, 255, 0), 
                                2
                            )
                        frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                    
                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                cap.release()
                os.unlink(video_path)
                progress_bar.empty()
            
            st.success(f"✅ Processed {frame_count} frames, {len(predictions)} predictions")
            
            df = pd.DataFrame(predictions, columns=['Frame', 'Prediction', 'Confidence'])
            df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.1%}")
            st.dataframe(df, use_container_width=True)
            
            valid_preds = [p for p in predictions if p[1] not in ["Low Confidence", "Error"]]
            if valid_preds:
                best_pred = max(valid_preds, key=lambda x: x[2])
                lang_code = 'ne' if tts_lang == "Nepali (नेपाली)" else 'en'
                
                if st.button("🔊 Speak Best Prediction", use_container_width=True):
                    audio = text_to_speech(best_pred[1], lang=lang_code)
                    if audio:
                        st.audio(audio, format='audio/mp3')
                        st.success(f"🔊 Speaking: {best_pred[1]}")

# ============================================================
# TAB 3: RESULTS
# ============================================================

with tab3:
    st.subheader("📊 Model Performance Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">99.9167%</div>
            <div class="metric-label">MobileNetV2 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">36</div>
            <div class="metric-label">NSL Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">7,200</div>
            <div class="metric-label">Validation Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">14</div>
            <div class="metric-label">Best Epoch</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("📋 Recognized NSL Classes")
    cols = st.columns(6)
    for i, cls in enumerate(NSL_CLASSES):
        cols[i % 6].markdown(f"✅ **{cls}**")
    
    st.subheader("📈 Training History")
    if os.path.exists("results/training_history.png"):
        st.image("results/training_history.png", use_container_width=True)
    elif os.path.exists("training_history.png"):
        st.image("training_history.png", use_container_width=True)
    else:
        st.info("Training history plot not found.")
    
    st.subheader("📊 Confusion Matrix")
    if os.path.exists("results/confusion_matrix.png"):
        st.image("results/confusion_matrix.png", use_container_width=True)
    elif os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", use_container_width=True)
    else:
        st.info("Confusion matrix not found.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>© 2026 NSL Pioneers | Himalayan AI Innovation Challenge | T6: NSL to Text/Speech</p>
    <p style="font-size: 0.8rem;">
        Baseline: MobileNetV2 (99.9167%) | Advanced: LSTM (Sequence Modeling) | TTS: gTTS
    </p>
</div>
""", unsafe_allow_html=True)