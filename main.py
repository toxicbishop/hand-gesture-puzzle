import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import numpy as np

WINDOW_NAME = "Live Puzzle"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)


def show_loading(message):
    splash = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(splash, "Live Puzzle", (430, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
    cv2.putText(splash, message, (430, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow(WINDOW_NAME, splash)
    cv2.waitKey(1)


show_loading("Loading hand tracker...")
from hand_tracker import HandTracker
from puzzle import Puzzle
from scores_manager import ScoresManager

# ================= CAMERA =================
show_loading("Opening camera...")

# CAP_DSHOW is significantly faster than the default MSMF backend on Windows.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Could not open camera. Trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open camera at all. Please check your connection.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

show_loading("Warming up model...")
tracker = HandTracker()
# Pre-warm MediaPipe with a dummy frame so the first real frame isn't slow.
tracker.find_hands(np.zeros((720, 1280, 3), dtype=np.uint8))
scores_handler = ScoresManager()

current_grid_size = 3
puzzle = Puzzle(current_grid_size)

mode = "camera"

# selection box
sel_x1 = sel_y1 = sel_x2 = sel_y2 = None

# states
start_time = None
end_time = None
solved = False

prev_pinch = False
dragging = False

# smoothing
smooth_x, smooth_y = 0, 0
alpha = 0.2

# trail
trail_points = []

# shuffle
shuffling = False
shuffle_start = 0

# reset timer
palm_hold_start = None
RESET_HOLD_TIME = 1.5 # seconds

# detection throttle (run MediaPipe every Nth frame)
frame_count = 0
DETECT_EVERY = 2


# ================= HELPERS =================
def inside_box(px, py, x1, y1, x2, y2, w, h):
    fx = px * w
    fy = py * h
    return x1 <= fx <= x2 and y1 <= fy <= y2


def to_local(px, py, x1, y1, x2, y2, w, h):
    fx = px * w
    fy = py * h

    if fx < x1 or fx > x2 or fy < y1 or fy > y2:
        return None

    lx = (fx - x1) / (x2 - x1)
    ly = (fy - y1) / (y2 - y1)

    return lx, ly


def draw_grid(img, x1, y1, x2, y2, rows=3, cols=3):
    cell_w = (x2 - x1) // cols
    cell_h = (y2 - y1) // rows

    for i in range(1, cols):
        cv2.line(img, (x1 + i*cell_w, y1), (x1 + i*cell_w, y2), (200, 200, 200), 1)

    for i in range(1, rows):
        cv2.line(img, (x1, y1 + i*cell_h), (x2, y1 + i*cell_h), (200, 200, 200), 1)


def draw_styled_text(img, text, pos, font_scale=0.8, color=(255, 255, 255), thickness=2):
    # Shadow
    cv2.putText(img, text, (pos[0]+2, pos[1]+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_overlay(img, text_lines, x, y, w, h):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    
    for i, line in enumerate(text_lines):
        draw_styled_text(img, line, (x + 20, y + 40 + i*40))


# ================= LOOP =================
while True:
    # ================= INPUT HANDLING =================
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'): break
    # Detect window closed via the X button
    try:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break
    if mode == "camera":
        if key == ord('1'): current_grid_size = 3
        if key == ord('2'): current_grid_size = 4
        if key == ord('3'): current_grid_size = 5

    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if frame_count % DETECT_EVERY == 0:
        tracker.find_hands(frame)
    tracker.draw_hands(frame)
    frame_count += 1

    pinch, px, py = tracker.get_pinch()
    detected, ix, iy = tracker.get_index_pos()
    two_hands, p1, p2 = tracker.get_two_hand_indices()
    palm_open = tracker.is_palm_open()

    # ================= RESET LOGIC =================
    if palm_open:
        if palm_hold_start is None:
            palm_hold_start = time.time()
        
        elapsed_hold = time.time() - palm_hold_start
        # Visual feedback for reset
        cv2.putText(frame, f"Resetting... {int((RESET_HOLD_TIME - elapsed_hold)*10)/10}s", 
                    (w//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if elapsed_hold > RESET_HOLD_TIME:
            mode = "camera"
            solved = False
            start_time = None
            palm_hold_start = None
    else:
        palm_hold_start = None

    # ================= CAMERA MODE =================
    if mode == "camera":
        best_time = scores_handler.get_best_time(current_grid_size)
        bt_text = f"Best: {best_time}s" if best_time != float('inf') else "Best: N/A"
        
        draw_styled_text(frame, f"Difficulty: {current_grid_size}x{current_grid_size}", (20, 40))
        draw_styled_text(frame, bt_text, (20, 80), color=(0, 255, 255))
        draw_styled_text(frame, "(Press 1, 2, 3 to change)", (20, 120), font_scale=0.5)

        if two_hands:
            x1, y1 = int(p1[0] * w), int(p1[1] * h)
            x2, y2 = int(p2[0] * w), int(p2[1] * h)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            color = (0, 0, 255) if pinch else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if pinch and not prev_pinch:
                if abs(x2 - x1) > 150 and abs(y2 - y1) > 150:
                    sel_x1, sel_y1, sel_x2, sel_y2 = x1, y1, x2, y2
                    crop = frame[y1:y2, x1:x2]
                    if crop.size != 0:
                        puzzle.create(crop, current_grid_size)
                        shuffling = True
                        shuffle_start = time.time()
                        mode = "puzzle"
                        solved = False

        prev_pinch = pinch
        cv2.imshow(WINDOW_NAME, frame)

    # ================= PUZZLE MODE =================
    else:
        output = frame.copy()

        if sel_x1 is not None:
            puzzle_img = puzzle.combine()
            puzzle_img = cv2.resize(puzzle_img, (sel_x2 - sel_x1, sel_y2 - sel_y1))
            output[sel_y1:sel_y2, sel_x1:sel_x2] = puzzle_img
            draw_grid(output, sel_x1, sel_y1, sel_x2, sel_y2, current_grid_size, current_grid_size)

        # pointer smoothing
        if detected:
            cx, cy = int(ix * w), int(iy * h)
            cx = max(sel_x1, min(cx, sel_x2))
            cy = max(sel_y1, min(cy, sel_y2))
            smooth_x = int(alpha * cx + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * cy + (1 - alpha) * smooth_y)

        # ===== SHUFFLE =====
        # The actual shuffle happens once in puzzle.create(); this is just a
        # short visual pause so the player can see the scramble before the
        # timer starts.
        if shuffling:
            if time.time() - shuffle_start < 1.0:
                draw_styled_text(output, "SHUFFLING...", (w//2 - 100, h//2))
            else:
                shuffling = False
                start_time = time.time()

        # ===== INTERACTION =====
        if not shuffling and not solved:
            if pinch and not prev_pinch:
                if inside_box(px, py, sel_x1, sel_y1, sel_x2, sel_y2, w, h):
                    local = to_local(px, py, sel_x1, sel_y1, sel_x2, sel_y2, w, h)
                    if local:
                        idx = puzzle.get_index(local[0], local[1])
                        puzzle.selected = idx
                        dragging = True
            elif not pinch and prev_pinch:
                if dragging and puzzle.selected is not None:
                    local = to_local(px, py, sel_x1, sel_y1, sel_x2, sel_y2, w, h)
                    if local:
                        idx2 = puzzle.get_index(local[0], local[1])
                        puzzle.swap(puzzle.selected, idx2)
                    puzzle.selected = None
                    dragging = False
                    if puzzle.is_solved():
                        solved = True
                        end_time = time.time()
                        final_time = end_time - start_time
                        scores_handler.update_score(current_grid_size, final_time)

        prev_pinch = pinch

        # Highlight the tile under the pointer while dragging
        target_idx = None
        if dragging and detected:
            local = to_local(px, py, sel_x1, sel_y1, sel_x2, sel_y2, w, h)
            if local:
                target_idx = puzzle.get_index(local[0], local[1])

        # Draw inside the puzzle region's coordinate space
        if sel_x1 is not None:
            puzzle_region = output[sel_y1:sel_y2, sel_x1:sel_x2]
            puzzle.draw_selected(puzzle_region, target_idx=target_idx)

        if not shuffling:
            puzzle.draw_reference(output)

        if detected:
            cv2.circle(output, (smooth_x, smooth_y), 8, (255, 255, 255), -1)
            cv2.circle(output, (smooth_x, smooth_y), 10, (200, 200, 200), 1)

        # ===== UI OVERLAYS =====
        if start_time and not solved:
            elapsed = time.time() - start_time
            draw_styled_text(output, f"Time: {elapsed:.1f}s", (20, 40), color=(0, 255, 255))
            
            perc = puzzle.get_solved_percentage()
            cv2.rectangle(output, (20, 60), (220, 80), (50, 50, 50), -1)
            cv2.rectangle(output, (20, 60), (20 + int(perc * 2), 80), (0, 255, 0), -1)
            draw_styled_text(output, f"{int(perc)}%", (230, 78), font_scale=0.5)

        if solved:
            final_t = end_time - start_time
            draw_overlay(output, [
                "PUZZLE SOLVED!",
                f"Your Time: {final_t:.2f}s",
                "Hold palm to play again"
            ], w//2 - 200, h//2 - 100, 400, 200)

        cv2.imshow(WINDOW_NAME, output)

cap.release()
cv2.destroyAllWindows()