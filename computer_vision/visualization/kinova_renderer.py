import cv2
import numpy as np
import math

class KinovaRenderer:
    def __init__(self):
        self.PRIMARY_TEAL = (95, 107, 0)      
        self.TEAL_DARK = (60, 70, 0)          
        self.GREEN = (0, 200, 100)            
        self.RED = (50, 50, 240)              
        self.YELLOW = (0, 180, 255)           
        self.ORANGE = (0, 160, 255)           # لون قوس الزاوية المشابه لـ Kemtai
        self.WHITE = (255, 255, 255)
        self.BLACK = (20, 20, 20)             
        self.GRAY_LIGHT = (180, 180, 180)     # لون الخطوط المتقطعة
        self.GRAY_TEXT = (190, 190, 190)

    def get_point(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    def rounded_rectangle(self, frame, x1, y1, x2, y2, color, radius=22, alpha=0.82):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def get_active_arm_indices(self, landmarks):
        """تحديد الذراع النشط تلقائياً بناءً على ارتفاع المعصم (اللي مرفوع أكتر هو اللي شغال)"""
        # 16 معصم يمين، 15 معصم يسار (الـ y الأقل يعني أعلى في الشاشة)
        if landmarks[16].y < landmarks[15].y:
            return 12, 14, 16, 24  # يمين (كتف، كوع، معصم، وسط)
        else:
            return 11, 13, 15, 23  # يسار

    def draw_dashed_line(self, img, pt1, pt2, color, thickness=2, dash_length=15):
        """رسم خطوط الجسم المتقطعة بشكل احترافي"""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0: return
        dashes = int(dist / dash_length)
        for i in range(dashes):
            if i % 2 == 0:  # ارسم قطعة وسيب قطعة
                start = i / dashes
                end = (i + 1) / dashes
                startX = int(x1 + (x2 - x1) * start)
                startY = int(y1 + (y2 - y1) * start)
                endX = int(x1 + (x2 - x1) * end)
                endY = int(y1 + (y2 - y1) * end)
                cv2.line(img, (startX, startY), (endX, endY), color, thickness, cv2.LINE_AA)

    def draw_skeleton(self, frame, results, exercise_type="Squat", is_correct=True):
        if not results.pose_landmarks:
            return frame

        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark
        
        is_arm_exercise = exercise_type in ["Arm Abduction", "Pushup", "Arm VW"]
        
        # لو تمرين ذراع، هنجيب المفاصل بتاعته عشان نرسمها Solid
        active_conns = []
        if is_arm_exercise:
            sh_idx, el_idx, wr_idx, _ = self.get_active_arm_indices(landmarks)
            active_conns = [(sh_idx, el_idx), (el_idx, wr_idx), (el_idx, sh_idx), (wr_idx, el_idx)]

        body_connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (23, 25), (25, 27),
            (24, 26), (26, 28)
        ]

        for p1, p2 in body_connections:
            if landmarks[p1].visibility > 0.2 and landmarks[p2].visibility > 0.2:
                pt1 = self.get_point(landmarks, p1, w, h)
                pt2 = self.get_point(landmarks, p2, w, h)
                
                if is_arm_exercise:
                    # لو الخط ده تبع الذراع النشط
                    if (p1, p2) in active_conns:
                        cv2.line(frame, pt1, pt2, self.WHITE, 4, cv2.LINE_AA)
                        cv2.circle(frame, pt1, 6, self.WHITE, -1, cv2.LINE_AA)
                        cv2.circle(frame, pt2, 6, self.WHITE, -1, cv2.LINE_AA)
                    else:
                        # باقي الجسم Dashed
                        self.draw_dashed_line(frame, pt1, pt2, self.GRAY_LIGHT, 2, 12)
                else:
                    # الرسم العادي لباقي التمارين زي السكوات
                    bone_color = self.GREEN if is_correct else self.RED
                    cv2.line(frame, pt1, pt2, self.BLACK, 6, cv2.LINE_AA)
                    cv2.line(frame, pt1, pt2, bone_color, 3, cv2.LINE_AA)
                    cv2.circle(frame, pt1, 5, self.WHITE, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt2, 5, self.WHITE, -1, cv2.LINE_AA)

        return frame

    def calculate_angle_points(self, p1, p2, p3):
        ang1 = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        ang2 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
        angle = abs(ang1 - ang2)
        if angle > 180:
            angle = 360 - angle
        return angle

    def draw_angle_arc(self, frame, p1, p2, p3, angle_value, color=(0, 160, 255), radius=45):
        ang1 = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        ang2 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
        
        start_angle = min(ang1, ang2)
        end_angle = max(ang1, ang2)
        
        if end_angle - start_angle > 180:
            start_angle, end_angle = end_angle, start_angle + 360

        overlay = frame.copy()
        cv2.ellipse(overlay, p2, (radius, radius), 0, start_angle, end_angle, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.ellipse(frame, p2, (radius, radius), 0, start_angle, end_angle, self.WHITE, 2, cv2.LINE_AA)
        cv2.circle(frame, p2, 8, self.WHITE, -1, cv2.LINE_AA)

    def draw_feedback_banner(self, frame, feedback_text):
        h, w, _ = frame.shape
        banner_w = int(w * 0.60)
        banner_h = 75
        banner_x = (w - banner_w) // 2
        banner_y = 20
        self.rounded_rectangle(frame, banner_x, banner_y, banner_x + banner_w, banner_y + banner_h, self.PRIMARY_TEAL, radius=20, alpha=0.88)
        cv2.rectangle(frame, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h), self.WHITE, 1, cv2.LINE_AA)
        text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = banner_x + (banner_w - text_size[0]) // 2
        text_y = banner_y + (banner_h + text_size[1]) // 2
        cv2.putText(frame, feedback_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2, cv2.LINE_AA)

    def draw_timer(self, frame, elapsed_seconds):
        elapsed_seconds = int(elapsed_seconds)
        center = (80, 110)
        cv2.circle(frame, center, 42, self.BLACK, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 42, self.PRIMARY_TEAL, 2, cv2.LINE_AA)
        text = str(elapsed_seconds)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        x = center[0] - size[0] // 2
        y = center[1] + 5
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.WHITE, 2, cv2.LINE_AA)
        sec_text = "Sec"
        sec_size = cv2.getTextSize(sec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        sec_x = center[0] - sec_size[0] // 2
        cv2.putText(frame, sec_text, (sec_x, center[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.GRAY_TEXT, 1, cv2.LINE_AA)

    def draw_rep_card(self, frame, rep_count, target_reps):
        h, w, _ = frame.shape
        card_x = int(w * 0.28)
        card_y = h - 95
        card_w = int(w * 0.24)
        card_h = 75
        self.rounded_rectangle(frame, card_x, card_y, card_x + card_w, card_y + card_h, self.BLACK, radius=18, alpha=0.85)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), self.PRIMARY_TEAL, 1, cv2.LINE_AA)
        rep_text = f"{rep_count} / {target_reps}"
        cv2.putText(frame, rep_text, (card_x + 75, card_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "Repetitions", (card_x + 75, card_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.GRAY_TEXT, 1, cv2.LINE_AA)
        cv2.circle(frame, (card_x + 35, card_y + 37), 12, self.PRIMARY_TEAL, -1, cv2.LINE_AA)
        progress = max(0.0, min(1.0, rep_count / max(target_reps, 1)))
        bar_x = card_x + 75
        bar_y = card_y + 44
        bar_w = card_w - 95
        bar_h = 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), self.PRIMARY_TEAL, -1)

    def draw_overlay(self, frame, state, rep_count, angles, last_prediction, performance_score=85, target_reps=10, elapsed_seconds=0):
        h, w, _ = frame.shape
        if performance_score < 60:
            feedback_text = "Adjust your posture. Keep your core tight."
        elif state in ["DESCENDING", "DOWN"]:
            feedback_text = "Control the movement on the way down."
        elif state in ["BOTTOM", "UP"]:
            feedback_text = "Hold the position. Good form."
        elif last_prediction == 0:
            feedback_text = "Incorrect form detected. Check your alignment."
        else:
            feedback_text = "Great form! Keep going."

        self.draw_feedback_banner(frame, feedback_text)
        self.draw_timer(frame, elapsed_seconds)
        self.draw_rep_card(frame, rep_count, target_reps)

        card_h = 75
        card_y = h - 95
        rep_x = int(w * 0.28)
        rep_w = int(w * 0.24)
        perf_x = rep_x + rep_w + 15
        perf_w = int(w * 0.18)
        performance_score = max(0, min(100, int(performance_score)))
        self.rounded_rectangle(frame, perf_x, card_y, perf_x + perf_w, card_y + card_h, self.BLACK, radius=18, alpha=0.85)
        border_color = self.GREEN if performance_score >= 80 else (self.YELLOW if performance_score >= 50 else self.RED)
        cv2.rectangle(frame, (perf_x, card_y), (perf_x + perf_w, card_y + card_h), border_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{performance_score}%", (perf_x + 20, card_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "Performance", (perf_x + 20, card_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.GRAY_TEXT, 1, cv2.LINE_AA)
        return frame