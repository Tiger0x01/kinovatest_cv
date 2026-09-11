import cv2
import pickle
import os

from pose.detector import PoseDetector
from pose.landmarks import LandmarkManager
from biomechanics.angles import get_squat_angles
from exercises.squat import SquatAnalyzer
from visualization.kinova_renderer import KinovaRenderer

def main():
    # 1. Initialize Modules
    detector = PoseDetector()
    landmark_manager = LandmarkManager()
    
    # Path relative to main.py inside computer_vision folder
    schema_path = os.path.join(os.path.dirname(__file__), '../models/feature_schema.json')
    model_path = os.path.join(os.path.dirname(__file__), '../models/squat_rf_model.pkl')
    
    squat_analyzer = SquatAnalyzer(schema_path=schema_path)
    renderer = KinovaRenderer()
    
    # 2. Load ML Model
    try:
        with open(model_path, 'rb') as f:
            rf_model = pickle.load(f)
        print("✅ Random Forest Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Model could not be loaded. ({e})")
        rf_model = None

    # 3. Start Camera
    cap = cv2.VideoCapture(0)
    last_prediction = None

    print("🚀 KINOVA Runtime Started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Mirror for user convenience
        h, w, _ = frame.shape

        # Step 1: Detect Pose
        results = detector.process_frame(frame)
        
        if results.pose_landmarks:
            # Step 2: Extract Landmarks
            joints = landmark_manager.extract_landmarks(results, w, h)
            
            # Step 3: Calculate Angles
            angles = get_squat_angles(joints)
            
            # Step 4: Analyze Squat Phase & Extract Features
            rep_completed, features = squat_analyzer.update(angles, joints)
            
            # Step 5: Predict Correct/Incorrect on Rep Completion
            if rep_completed and features is not None:
                print(f"Rep Completed! Features shape: {features.shape}")
                if rf_model:
                    # Expected output: 1 (Correct) or 0 (Incorrect)
                    prediction = rf_model.predict(features)
                    last_prediction = prediction[0]
                    print(f"Model Prediction: {last_prediction}")

            # Step 6: Rendering
            frame = renderer.draw_skeleton(frame, results)
            frame = renderer.draw_overlay(
                frame, 
                state=squat_analyzer.state, 
                rep_count=squat_analyzer.rep_count, 
                angles=angles, 
                last_prediction=last_prediction
            )

        cv2.imshow('KINOVA Squat Analysis', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()