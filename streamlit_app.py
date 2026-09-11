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

run_app = st.checkbox("Start Camera", key="run_cam")
FRAME_WINDOW = st.image([])
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

detector, landmark_manager, analyzer, renderer, ml_model = load_architecture(selected_exercise)

if run_app:
    # إعادة ضبط الإحصائيات عند بدء تمرين جديد
    st.session_state.session_stats = {'reps': 0, 'perf_scores': [], 'duration': 0}
    REPORT_PLACEHOLDER.empty()
    
    cap = cv2.VideoCapture(0)
    last_prediction = None
    start_time = time.time()  
    
    st.session_state.arm_reps = 0
    st.session_state.arm_state = "DOWN"

    while st.session_state.run_cam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        results = detector.process_frame(frame)
        elapsed_time = time.time() - start_time
        current_perf = 85 # قيمة افتراضية لحين تحديث الأداء
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            joints = landmark_manager.extract_landmarks(results, w, h)
            
            # ====== منطق تمارين الذراع ======
            if selected_exercise in ["Arm Abduction", "Pushup", "Arm VW"]:
                # 1. تحديد الذراع النشط تلقائياً
                sh_idx, el_idx, wr_idx, hip_idx = renderer.get_active_arm_indices(landmarks)
                
                p_hip = renderer.get_point(landmarks, hip_idx, w, h)
                p_sh = renderer.get_point(landmarks, sh_idx, w, h)
                p_el = renderer.get_point(landmarks, el_idx, w, h)
                p_wr = renderer.get_point(landmarks, wr_idx, w, h)
                
                # حساب زاوية الكتف (للـ Abduction)
                arm_angle = renderer.calculate_angle_points(p_hip, p_sh, p_el)
                
                # 2. عداد التمارين للذراع (State Machine)
                if st.session_state.arm_state == "DOWN":
                    if arm_angle > 65:  # رفع الذراع
                        st.session_state.arm_state = "UP"
                elif st.session_state.arm_state == "UP":
                    if arm_angle < 35:  # إنزال الذراع واكتمال العدة
                        st.session_state.arm_state = "DOWN"
                        st.session_state.arm_reps += 1
                        st.session_state.session_stats['reps'] = st.session_state.arm_reps
                
                current_reps = st.session_state.arm_reps
                current_state = st.session_state.arm_state
                
                # 3. الرسم (الجسم داشد، الذراع بارز، القوس البرتقالي)
                frame = renderer.draw_skeleton(frame, results, exercise_type=selected_exercise)
                renderer.draw_angle_arc(frame, p_hip, p_sh, p_el, arm_angle, color=renderer.ORANGE)

            # ====== منطق السكوات والتمارين العادية ======
            else:
                angles = get_squat_angles(joints)
                rep_completed, features = analyzer.update(angles, joints)
                
                if rep_completed:
                    st.session_state.session_stats['reps'] = analyzer.rep_count
                    
                current_reps = analyzer.rep_count
                current_state = analyzer.state
                
                # الرسم العادي
                frame = renderer.draw_skeleton(frame, results, exercise_type=selected_exercise, is_correct=True)

            # رسم الواجهة النهائية (العدادات والنصائح)
            frame = renderer.draw_overlay(
                frame=frame, 
                state=current_state, 
                rep_count=current_reps, 
                angles=None, 
                last_prediction=last_prediction,
                performance_score=current_perf,
                target_reps=target_reps,
                elapsed_seconds=elapsed_time
            )
            
        st.session_state.session_stats['duration'] = int(elapsed_time)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
    cap.release()

# 2. عرض التقرير النهائي (Summary Report) عند إيقاف الكاميرا
elif not run_app and st.session_state.session_stats['reps'] > 0:
    stats = st.session_state.session_stats
    avg_perf = int(np.mean(stats['perf_scores'])) if stats['perf_scores'] else 0
    
    color = "#00C864" if avg_perf > 75 else ("#FFB400" if avg_perf > 50 else "#FF3232")
    
    report_html = f"""
    <div style="background-color: #f8f9fa; border-radius: 20px; padding: 40px; text-align: center; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); margin-top: 20px;">
        <h2 style="color: #333; font-family: sans-serif;">Summary</h2>
        <h1 style="color: #333; font-size: 2.5rem; margin-bottom: 30px;">{selected_exercise}</h1>
        
        <div style="display: flex; justify-content: center; align-items: center; gap: 50px;">
            <div style="border: 8px solid {color}; border-radius: 50%; width: 200px; height: 200px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <span style="font-size: 1.2rem; color: #666;">Performance</span>
                <span style="font-size: 3.5rem; font-weight: bold; color: {color};">{avg_perf}%</span>
            </div>
            
            <div style="text-align: left;">
                <h3 style="color: #555; margin-bottom: 10px;">⏱️ Duration: <span style="color: #000;">{stats['duration']} Seconds</span></h3>
                <h3 style="color: #555;">🔄 Repetitions: <span style="color: #000;">{stats['reps']} / {target_reps}</span></h3>
            </div>
        </div>
    </div>
    """
    REPORT_PLACEHOLDER.markdown(report_html, unsafe_allow_html=True)
else:
    st.info("👈 Please select an exercise from the sidebar and click 'Start Camera' to begin.")