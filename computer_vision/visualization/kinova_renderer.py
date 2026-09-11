import cv2
import numpy as np

class KinovaRenderer:
    def __init__(self):
        # 🎨 Palettes & Professional Color System (BGR)
        self.PRIMARY_TEAL = (95, 107, 0)      # #006B5F (Petroleum / Teal احترافي)
        self.TEAL_DARK = (60, 70, 0)          
        self.GREEN = (0, 200, 100)            # أخضر مريح للنجاح والوضع الصحيح
        self.RED = (50, 50, 240)              # أحمر ناعم للتنبيهات والخطأ
        self.YELLOW = (0, 180, 255)           # أصفر/برتقالي هادئ للتحذيرات
        self.WHITE = (255, 255, 255)
        self.BLACK = (20, 20, 20)             # أسود فحمي مريح للعين
        self.GRAY_LIGHT = (240, 240, 240)
        self.GRAY_TEXT = (190, 190, 190)

    def get_point(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    def rounded_rectangle(self, frame, x1, y1, x2, y2, color, radius=22, alpha=0.82):
        """رسم بطاقة أو مستطيل عائم ذو زوايا دائرية وشفافية ناعمة."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_skeleton(self, frame, results, angles=None, is_correct=True):
        """رسم الهيكل العظمي والمفاصل بتصميم نظيف ومتناسق (بدون زوايا)."""
        if not results.pose_landmarks:
            return frame

        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        bone_color = self.GREEN if is_correct else self.RED
        joint_color = self.WHITE

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
                cv2.line(frame, pt1, pt2, self.BLACK, 6, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, bone_color, 3, cv2.LINE_AA)

        body_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for idx in body_joints:
            if landmarks[idx].visibility > 0.2:
                pt = self.get_point(landmarks, idx, w, h)
                cv2.circle(frame, pt, 8, self.BLACK, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)

        return frame

    def draw_feedback_banner(self, frame, feedback_text):
        """شريط التوجيه العلوي بلون البترولي الاحترافي (#006B5F)."""
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
        """عداد الوقت في أعلى اليسار."""
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
        """بطاقة العداد السفلية مع شريط التقدم."""
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
        """التنسيق الشامل للواجهة بالكامل."""
        h, w, _ = frame.shape

        if state == "DESCENDING":
            feedback_text = "Spread your feet a little wider"
        elif state == "BOTTOM":
            feedback_text = "Good form - keep your back straight"
        elif last_prediction == 0:
            feedback_text = "Incorrect form - adjust your position"
        else:
            feedback_text = "Great form - keep going"

        self.draw_feedback_banner(frame, feedback_text)
        self.draw_timer(frame, elapsed_seconds)

        card_h = 75
        card_y = h - 95
        rep_x = int(w * 0.28)
        rep_w = int(w * 0.24)
        perf_x = rep_x + rep_w + 15
        perf_w = int(w * 0.18)

        self.draw_rep_card(frame, rep_count, target_reps)

        performance_score = max(0, min(100, int(performance_score)))
        self.rounded_rectangle(frame, perf_x, card_y, perf_x + perf_w, card_y + card_h, self.BLACK, radius=18, alpha=0.85)
        
        border_color = self.GREEN if performance_score >= 80 else (self.YELLOW if performance_score >= 50 else self.RED)
        cv2.rectangle(frame, (perf_x, card_y), (perf_x + perf_w, card_y + card_h), border_color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"{performance_score}%", (perf_x + 20, card_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "Performance", (perf_x + 20, card_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.GRAY_TEXT, 1, cv2.LINE_AA)

        return frame