import numpy as np
import json

class SquatAnalyzer:
    def __init__(self, schema_path='../models/feature_schema.json'):
        self.state = 'STANDING'
        self.rep_count = 0
        self.frame_history = []
        
        try:
            with open(schema_path, 'r') as f:
                self.feature_names = json.load(f)
        except:
            self.feature_names = [f"feature_{i}" for i in range(29)]
            
    def update(self, angles, joints):
        # 1. أخذ أقل زاويتي ركبة (اليمين أو اليسار) لضمان الاستجابة بغض النظر عن اتجاه وقوفك أمام الكاميرا
        r_knee = angles.get('right_knee', 180)
        l_knee = angles.get('left_knee', 180)
        knee_angle = min(r_knee, l_knee)  # الأقرب للثني (الأساس في السكوات)
        
        self.frame_history.append(angles)
        rep_completed = False
        features = None
        
        # 2. State Machine مرنة جداً ومضبوطة للتجربة الحية
        if self.state == 'STANDING':
            if knee_angle < 165:  # بدء الانحناء البسيط
                self.state = 'DESCENDING'
                self.frame_history = [angles]
                
        elif self.state == 'DESCENDING':
            if knee_angle < 130:   # الوصول لمنطقة النزول الفعلي
                self.state = 'BOTTOM'
                
        elif self.state == 'BOTTOM':
            if knee_angle > 135:  # بدء الصعود
                self.state = 'ASCENDING'
                
        elif self.state == 'ASCENDING':
            if knee_angle > 165:  # العودة للوقوف واكتمال العدة بنجاح
                self.state = 'STANDING'
                self.rep_count += 1
                rep_completed = True
                features = self.extract_features()
                self.frame_history = []
                
        return rep_completed, features

    def extract_features(self):
        if not self.frame_history:
            return np.zeros(29)
            
        r_knee_arr = np.array([f.get('right_knee', 180) for f in self.frame_history])
        l_knee_arr = np.array([f.get('left_knee', 180) for f in self.frame_history])
        r_hip_arr = np.array([f.get('right_hip', 180) for f in self.frame_history])
        r_trunk_arr = np.array([f.get('right_trunk', 0) for f in self.frame_history])
        
        raw_features = {
            'r_knee_min': np.min(r_knee_arr),
            'r_knee_max': np.max(r_knee_arr),
            'r_knee_rom': np.max(r_knee_arr) - np.min(r_knee_arr),
            'r_knee_mean': np.mean(r_knee_arr),
            'r_knee_std': np.std(r_knee_arr),
            'l_knee_min': np.min(l_knee_arr),
            'l_knee_rom': np.max(l_knee_arr) - np.min(l_knee_arr),
            'r_hip_min': np.min(r_hip_arr),
            'r_hip_max': np.max(r_hip_arr),
            'r_hip_rom': np.max(r_hip_arr) - np.min(r_hip_arr),
            'trunk_max': np.max(r_trunk_arr),
            'trunk_mean': np.mean(r_trunk_arr),
            'rep_duration': len(self.frame_history)
        }
        
        feature_vector = np.zeros(29)
        for i, feature_name in enumerate(self.feature_names):
            feature_vector[i] = raw_features.get(feature_name, 0.0)
            
        return feature_vector.reshape(1, -1)