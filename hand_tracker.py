import cv2
import mediapipe as mp
import math

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # 🔥 MUST BE 2
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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
                x1, y1 = hand.landmark[4].x, hand.landmark[4].y  # thumb
                x2, y2 = hand.landmark[8].x, hand.landmark[8].y  # index

                dist = ((x2-x1)**2 + (y2-y1)**2)**0.5

                # 🔥 tighter condition
                if dist < 0.05:
                    return True, x2, y2

            # no pinch
            hand = self.results.multi_hand_landmarks[0]
            return False, hand.landmark[8].x, hand.landmark[8].y

        return False, 0, 0

    def get_two_fingers(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]

            index_up = hand.landmark[8].y < hand.landmark[6].y
            middle_up = hand.landmark[12].y < hand.landmark[10].y

            if index_up and middle_up:
                x = hand.landmark[8].x
                y = hand.landmark[8].y
                return True, x, y

        return False, 0, 0
    def get_two_fingers(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]

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
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            x = hand.landmark[8].x
            y = hand.landmark[8].y
            return True, x, y

        return False, 0, 0
    def get_two_hand_positions(self):
        points = []

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                x = hand.landmark[8].x
                y = hand.landmark[8].y
                points.append((x, y))

        return points