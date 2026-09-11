import streamlit as st
import cv2
import pickle
import sys
import os
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'computer_vision'))

from pose.detector import PoseDetector
from pose.landmarks import LandmarkManager
from biomechanics.angles import get_squat_angles
from exercises.squat import SquatAnalyzer
from visualization.kinova_renderer import KinovaRenderer

st.set_page_config(page_title="KINOVA - Squat Analyzer", layout="wide")

st.title("🦾 KINOVA: AI Biomechanics Analysis")
st.markdown("Real-time Squat tracking using MediaPipe and Random Forest.")

run_app = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

# Load Architecture
@st.cache_resource
def load_architecture():
    detector = PoseDetector()
    landmark_manager = LandmarkManager()
    squat_analyzer = SquatAnalyzer(schema_path='models/feature_schema.json')
    renderer = KinovaRenderer()
    
    try:
        with open('models/squat_rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
    except:
        rf_model = None
        
    return detector, landmark_manager, squat_analyzer, renderer, rf_model

detector, landmark_manager, squat_analyzer, renderer, rf_model = load_architecture()

if run_app:
    cap = cv2.VideoCapture(0)
    last_prediction = None
    start_time = time.time()  
    
    while run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Pipeline
        results = detector.process_frame(frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            joints = landmark_manager.extract_landmarks(results, w, h)
            angles = get_squat_angles(joints)
            
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

            left_ankle = np.array([landmarks[27].x, landmarks[27].y])
            right_ankle = np.array([landmarks[28].x, landmarks[28].y])
            foot_distance = np.linalg.norm(left_ankle - right_ankle)

            target_foot_distance = shoulder_width * 1.35
            error = abs(foot_distance - target_foot_distance)
            performance_score = int(max(0, min(100, 100 - (error / (shoulder_width + 1e-6) * 120))))

            rep_completed, features = squat_analyzer.update(angles, joints)
            
            if rep_completed and features is not None and rf_model:
                last_prediction = rf_model.predict(features)[0]
                
            elapsed_time = time.time() - start_time
                
            frame = renderer.draw_skeleton(frame, results, angles, is_correct=(performance_score > 50))
            frame = renderer.draw_overlay(
                frame=frame, 
                state=squat_analyzer.state, 
                rep_count=squat_analyzer.rep_count, 
                angles=angles, 
                last_prediction=last_prediction,
                performance_score=performance_score,
                target_reps=10,
                elapsed_seconds=elapsed_time
            )
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
else:
    st.write("Check 'Start Camera' to begin KINOVA runtime.")