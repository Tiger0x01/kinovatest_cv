import streamlit as st
import cv2
import pickle
import sys
import os
import time
import numpy as np
import av
import threading

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

sys.path.append(os.path.join(os.path.dirname(__file__), 'computer_vision'))

from pose.detector import PoseDetector
from pose.landmarks import LandmarkManager
from biomechanics.angles import get_squat_angles  # ستحتاج لإضافة دوال حساب زوايا الذراع لاحقاً
from exercises.squat import SquatAnalyzer
from visualization.kinova_renderer import KinovaRenderer

st.set_page_config(page_title="KINOVA - AI Biomechanics", layout="wide")

# 1. إعدادات القائمة الجانبية (Sidebar)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/81/Artificial_Intelligence_AI.svg", width=100) # يمكنك وضع لوجو KINOVA هنا
st.sidebar.title("⚙️ KINOVA Settings")

# خريطة النماذج بناءً على هيكل الملفات لديك
EXERCISE_MAP = {
    "Squat": {"model": "models/squat/squat_rf_model.pkl", "schema": "models/squat/feature_schema.json"},
    "Arm Abduction": {"model": "models/armabduction/arm_abduction_knn.pkl", "schema": "models/armabduction/arm_abduction_feature_schema.json"},
    "Pushup": {"model": "models/pushup/pushup_rf_model.pkl", "schema": "models/pushup/pushup_feature_schema.json"},
    "Lunge": {"model": "models/lunge/lunge_knn_model.pkl", "schema": "models/lunge/lunge_feature_schema.json"},
    "Arm VW": {"model": "models/armvw/arm_vw_svm_rbf.pkl", "schema": "models/armvw/arm_vw_feature_schema.json"},
    "Leg Abduction": {"model": "models/legabduction/leg_abduction_svm_linear.pkl", "schema": "models/legabduction/leg_abduction_feature_schema.json"},
}

selected_exercise = st.sidebar.selectbox("🎯 Select Exercise", list(EXERCISE_MAP.keys()))
target_reps = st.sidebar.number_input("Target Repetitions", min_value=1, max_value=50, value=10)

st.title(f"🦾 KINOVA: {selected_exercise} Analysis")
st.markdown("Real-time biomechanics tracking using MediaPipe and ML Models.")

REPORT_PLACEHOLDER = st.empty()

# تهيئة متغيرات الجلسة للتقرير النهائي
if 'session_stats' not in st.session_state:
    st.session_state.session_stats = {'reps': 0, 'perf_scores': [], 'duration': 0}

@st.cache_resource
def load_architecture(exercise_name):
    detector = PoseDetector()
    landmark_manager = LandmarkManager()
    renderer = KinovaRenderer()
    
    schema_path = EXERCISE_MAP[exercise_name]["schema"]
    model_path = EXERCISE_MAP[exercise_name]["model"]
    
    # ملاحظة: حالياً نستخدم SquatAnalyzer كمثال. ستحتاج لإنشاء Analyzer خاص بكل تمرين.
    analyzer = SquatAnalyzer(schema_path=schema_path) 
    
    try:
        with open(model_path, 'rb') as f:
            ml_model = pickle.load(f)
    except:
        ml_model = None
        
    return detector, landmark_manager, analyzer, renderer, ml_model

class KINOVAVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.detector, self.landmark_manager, self.analyzer, self.renderer, self.ml_model = (
            load_architecture(selected_exercise)
        )

        self.exercise = selected_exercise
        self.target_reps = target_reps

        self.arm_reps = 0
        self.arm_state = "DOWN"
        self.start_time = time.time()
        self.rep_count = 0

        self.lock = threading.Lock()

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Mirror camera
        img = cv2.flip(img, 1)

        h, w, _ = img.shape

        # MediaPipe
        results = self.detector.process_frame(img)

        elapsed_time = time.time() - self.start_time

        current_perf = 85
        last_prediction = None

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            joints = self.landmark_manager.extract_landmarks(
                results,
                w,
                h
            )

            # -----------------------------
            # ARM EXERCISES
            # -----------------------------

            if self.exercise in [
                "Arm Abduction",
                "Pushup",
                "Arm VW"
            ]:

                sh_idx, el_idx, wr_idx, hip_idx = (
                    self.renderer.get_active_arm_indices(
                        landmarks
                    )
                )

                p_hip = self.renderer.get_point(
                    landmarks,
                    hip_idx,
                    w,
                    h
                )

                p_sh = self.renderer.get_point(
                    landmarks,
                    sh_idx,
                    w,
                    h
                )

                p_el = self.renderer.get_point(
                    landmarks,
                    el_idx,
                    w,
                    h
                )

                p_wr = self.renderer.get_point(
                    landmarks,
                    wr_idx,
                    w,
                    h
                )

                arm_angle = self.renderer.calculate_angle_points(
                    p_hip,
                    p_sh,
                    p_el
                )

                if self.arm_state == "DOWN":

                    if arm_angle > 65:
                        self.arm_state = "UP"

                elif self.arm_state == "UP":

                    if arm_angle < 35:

                        self.arm_state = "DOWN"

                        self.arm_reps += 1

                current_reps = self.arm_reps
                current_state = self.arm_state

                img = self.renderer.draw_skeleton(
                    img,
                    results,
                    exercise_type=self.exercise
                )

                self.renderer.draw_angle_arc(
                    img,
                    p_hip,
                    p_sh,
                    p_el,
                    arm_angle,
                    color=self.renderer.ORANGE
                )

            # -----------------------------
            # OTHER EXERCISES
            # -----------------------------

            else:

                angles = get_squat_angles(joints)

                rep_completed, features = self.analyzer.update(
                    angles,
                    joints
                )

                current_reps = self.analyzer.rep_count
                current_state = self.analyzer.state

                img = self.renderer.draw_skeleton(
                    img,
                    results,
                    exercise_type=self.exercise,
                    is_correct=True
                )

            # -----------------------------
            # OVERLAY
            # -----------------------------

            img = self.renderer.draw_overlay(
                frame=img,
                state=current_state,
                rep_count=current_reps,
                angles=None,
                last_prediction=last_prediction,
                performance_score=current_perf,
                target_reps=self.target_reps,
                elapsed_seconds=elapsed_time
            )

            self.rep_count = current_reps

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )
st.subheader("Live Camera")

rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)

ctx = webrtc_streamer(
    key="kinova-camera",
    video_processor_factory=KINOVAVideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
if ctx.state.playing:
    st.success("Camera is running")
else:
    st.info("Click START to open your camera.")

