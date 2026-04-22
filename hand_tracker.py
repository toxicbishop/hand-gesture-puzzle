import cv2
import mediapipe as mp
import math

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

    def draw_hands(self, frame):
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, handLms, self.mp_hands.HAND_CONNECTIONS
                )

    def get_pinch(self):
        if self.results and self.results.multi_hand_landmarks:

            for hand in self.results.multi_hand_landmarks:
                x1, y1 = hand.landmark[4].x, hand.landmark[4].y  # thumb tip
                x2, y2 = hand.landmark[8].x, hand.landmark[8].y  # index tip

                # Normalize by hand size (wrist -> index MCP) so the threshold
                # is robust to how far the hand is from the camera.
                wx, wy = hand.landmark[0].x, hand.landmark[0].y
                mx, my = hand.landmark[5].x, hand.landmark[5].y
                hand_size = ((mx-wx)**2 + (my-wy)**2)**0.5 or 1e-6

                dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
                if dist / hand_size < 0.35:
                    return True, x2, y2

            hand = self.results.multi_hand_landmarks[0]
            return False, hand.landmark[8].x, hand.landmark[8].y

        return False, 0, 0

    def get_two_fingers(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]

            index_up = hand.landmark[8].y < hand.landmark[6].y
            middle_up = hand.landmark[12].y < hand.landmark[10].y

            ix, iy = hand.landmark[8].x, hand.landmark[8].y   # index
            mx, my = hand.landmark[12].x, hand.landmark[12].y # middle

            return True, ix, iy, mx, my

        return False, 0, 0, 0, 0
    
    def get_two_hand_indices(self):
        points = []

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                x = hand.landmark[8].x
                y = hand.landmark[8].y
                points.append((x, y))

        if len(points) >= 2:
            return True, points[0], points[1]

        return False, (0, 0), (0, 0)

    def get_index_pos(self):
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            x = hand.landmark[8].x
            y = hand.landmark[8].y
            return True, x, y
        return False, 0, 0

    def is_palm_open(self):
        """Detects if all fingers are extended (open palm)."""
        if not self.results or not self.results.multi_hand_landmarks:
            return False
            
        for hand in self.results.multi_hand_landmarks:
            # Indices for fingertips and their corresponding MCP (base) joints
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]
            
            fingers_up = []
            for tip, pip in zip(tips, pips):
                if hand.landmark[tip].y < hand.landmark[pip].y:
                    fingers_up.append(True)
                else:
                    fingers_up.append(False)
            
            # Thumb: tip x vs ip x (depends on hand orientation, but simple distance often works)
            # For simplicity, we check if all 4 main fingers are up
            if all(fingers_up):
                return True
        return False

    def get_two_hand_positions(self):
        points = []
        if self.results and self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                x = hand.landmark[8].x
                y = hand.landmark[8].y
                points.append((x, y))
        return points