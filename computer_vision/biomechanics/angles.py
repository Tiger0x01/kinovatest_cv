import numpy as np

def calculate_angle_2d(p1, p2, p3):
    """
    حساب الزاوية بدقة باستخدام الـ Dot Product لمنع أخطاء الـ Wrap-around (مثل مشكلة 2 درجة).
    p2 هي نقطة الرأس (Vertex).
    """
    a = np.array([p1['x'], p1['y']])
    b = np.array([p2['x'], p2['y']])
    c = np.array([p3['x'], p3['y']])
    
    ba = a - b
    bc = c - b
    
    # حساب جيب تمام الزاوية
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    
    return angle

def calculate_trunk_angle(shoulder, hip):
    vertical_pt = {'x': hip['x'], 'y': hip['y'] - 0.1}
    return calculate_angle_2d(shoulder, hip, vertical_pt)

def get_squat_angles(joints):
    angles = {}
    
    # Left Side
    angles['left_knee'] = calculate_angle_2d(joints['left_hip'], joints['left_knee'], joints['left_ankle'])
    angles['left_hip'] = calculate_angle_2d(joints['left_shoulder'], joints['left_hip'], joints['left_knee'])
    angles['left_trunk'] = calculate_trunk_angle(joints['left_shoulder'], joints['left_hip'])
    
    # Right Side
    angles['right_knee'] = calculate_angle_2d(joints['right_hip'], joints['right_knee'], joints['right_ankle'])
    angles['right_hip'] = calculate_angle_2d(joints['right_shoulder'], joints['right_hip'], joints['right_knee'])
    angles['right_trunk'] = calculate_trunk_angle(joints['right_shoulder'], joints['right_hip'])
    
    return angles