import mediapipe as mp

class LandmarkManager:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def extract_landmarks(self, results, frame_width, frame_height):
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        
        def get_lm(mp_enum):
            lm = landmarks[mp_enum.value]
            return {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'vis': lm.visibility,
                'px': int(lm.x * frame_width),
                'py': int(lm.y * frame_height)
            }

        joints = {
            'left_shoulder': get_lm(self.mp_pose.PoseLandmark.LEFT_SHOULDER),
            'right_shoulder': get_lm(self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            'left_hip': get_lm(self.mp_pose.PoseLandmark.LEFT_HIP),
            'right_hip': get_lm(self.mp_pose.PoseLandmark.RIGHT_HIP),
            'left_knee': get_lm(self.mp_pose.PoseLandmark.LEFT_KNEE),
            'right_knee': get_lm(self.mp_pose.PoseLandmark.RIGHT_KNEE),
            'left_ankle': get_lm(self.mp_pose.PoseLandmark.LEFT_ANKLE),
            'right_ankle': get_lm(self.mp_pose.PoseLandmark.RIGHT_ANKLE),
        }
        
        return joints